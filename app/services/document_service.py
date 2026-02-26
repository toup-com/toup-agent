"""
Document Processing Service for HexBrain

Handles:
- PDF extraction
- Markdown parsing
- Code file processing
- Text chunking for embeddings
- Image description (via AI)
- Video/Audio transcription (via AI)
"""

import os
import io
import re
import json
import uuid
import hashlib
import mimetypes
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

# Document processing
try:
    from pypdf import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False


@dataclass
class ExtractedChunk:
    """A chunk of text extracted from a document"""
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DocumentExtraction:
    """Result of document extraction"""
    text: str
    chunks: List[ExtractedChunk]
    page_count: Optional[int] = None
    word_count: int = 0
    language: Optional[str] = None
    encoding: Optional[str] = None
    programming_language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MediaExtraction:
    """Result of media extraction"""
    ai_description: Optional[str] = None
    ai_transcript: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    format: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentService:
    """Service for processing documents and media files"""
    
    # File type detection
    DOCUMENT_EXTENSIONS = {
        '.pdf': 'pdf',
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.txt': 'text',
        '.text': 'text',
        '.py': 'code',
        '.js': 'code',
        '.ts': 'code',
        '.jsx': 'code',
        '.tsx': 'code',
        '.java': 'code',
        '.cpp': 'code',
        '.c': 'code',
        '.h': 'code',
        '.go': 'code',
        '.rs': 'code',
        '.rb': 'code',
        '.php': 'code',
        '.swift': 'code',
        '.kt': 'code',
        '.scala': 'code',
        '.r': 'code',
        '.sql': 'code',
        '.sh': 'code',
        '.bash': 'code',
        '.zsh': 'code',
        '.ps1': 'code',
        '.docx': 'docx',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.csv': 'csv',
        '.html': 'text',
        '.htm': 'text',
        '.xml': 'text',
    }
    
    MEDIA_EXTENSIONS = {
        '.jpg': 'image',
        '.jpeg': 'image',
        '.png': 'image',
        '.gif': 'image',
        '.webp': 'image',
        '.bmp': 'image',
        '.svg': 'image',
        '.mp4': 'video',
        '.mov': 'video',
        '.avi': 'video',
        '.webm': 'video',
        '.mkv': 'video',
        '.mp3': 'audio',
        '.wav': 'audio',
        '.ogg': 'audio',
        '.flac': 'audio',
        '.m4a': 'audio',
    }
    
    PROGRAMMING_LANGUAGES = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.sql': 'sql',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'zsh',
        '.ps1': 'powershell',
    }
    
    def __init__(
        self,
        embedding_service=None,
        openai_client=None,
        storage_path: str = "/tmp/hexbrain_uploads"
    ):
        self.embedding_service = embedding_service
        self.openai_client = openai_client
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def detect_file_type(self, filename: str, content: bytes = None) -> Tuple[str, str]:
        """
        Detect file type from filename and content.
        Returns (file_type, mime_type)
        """
        ext = os.path.splitext(filename.lower())[1]
        
        # Check document types
        if ext in self.DOCUMENT_EXTENSIONS:
            file_type = self.DOCUMENT_EXTENSIONS[ext]
            mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            return file_type, mime_type
        
        # Check media types
        if ext in self.MEDIA_EXTENSIONS:
            file_type = self.MEDIA_EXTENSIONS[ext]
            mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            return file_type, mime_type
        
        # Fallback to text
        return 'text', 'text/plain'
    
    def compute_file_hash(self, content: bytes) -> str:
        """Compute SHA256 hash of file content"""
        return hashlib.sha256(content).hexdigest()
    
    async def extract_document(
        self,
        content: bytes,
        filename: str,
        file_type: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> DocumentExtraction:
        """
        Extract text and chunks from a document.
        """
        ext = os.path.splitext(filename.lower())[1]
        
        if file_type == 'pdf':
            return await self._extract_pdf(content, chunk_size, chunk_overlap)
        elif file_type == 'docx':
            return await self._extract_docx(content, chunk_size, chunk_overlap)
        elif file_type == 'markdown':
            return await self._extract_markdown(content, chunk_size, chunk_overlap)
        elif file_type == 'code':
            return await self._extract_code(content, ext, chunk_size, chunk_overlap)
        elif file_type == 'json':
            return await self._extract_json(content, chunk_size, chunk_overlap)
        elif file_type == 'yaml':
            return await self._extract_yaml(content, chunk_size, chunk_overlap)
        elif file_type == 'csv':
            return await self._extract_csv(content, chunk_size, chunk_overlap)
        else:
            return await self._extract_text(content, chunk_size, chunk_overlap)
    
    async def _extract_pdf(
        self,
        content: bytes,
        chunk_size: int,
        chunk_overlap: int
    ) -> DocumentExtraction:
        """Extract text from PDF"""
        if not HAS_PDF:
            raise ValueError("PDF support not available. Install pypdf.")
        
        reader = PdfReader(io.BytesIO(content))
        pages_text = []
        
        for page in reader.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
        
        full_text = "\n\n".join(pages_text)
        chunks = self._chunk_text_with_pages(pages_text, chunk_size, chunk_overlap)
        
        return DocumentExtraction(
            text=full_text,
            chunks=chunks,
            page_count=len(reader.pages),
            word_count=len(full_text.split()),
            metadata={"pdf_pages": len(reader.pages)}
        )
    
    async def _extract_docx(
        self,
        content: bytes,
        chunk_size: int,
        chunk_overlap: int
    ) -> DocumentExtraction:
        """Extract text from DOCX"""
        if not HAS_DOCX:
            raise ValueError("DOCX support not available. Install python-docx.")
        
        doc = DocxDocument(io.BytesIO(content))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        full_text = "\n\n".join(paragraphs)
        chunks = self._chunk_text(full_text, chunk_size, chunk_overlap)
        
        return DocumentExtraction(
            text=full_text,
            chunks=chunks,
            word_count=len(full_text.split()),
            metadata={"paragraph_count": len(paragraphs)}
        )
    
    async def _extract_markdown(
        self,
        content: bytes,
        chunk_size: int,
        chunk_overlap: int
    ) -> DocumentExtraction:
        """Extract text from Markdown"""
        text = self._decode_bytes(content)
        
        # Split by headers for better chunking
        chunks = self._chunk_markdown(text, chunk_size, chunk_overlap)
        
        return DocumentExtraction(
            text=text,
            chunks=chunks,
            word_count=len(text.split()),
            metadata={"format": "markdown"}
        )
    
    async def _extract_code(
        self,
        content: bytes,
        ext: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> DocumentExtraction:
        """Extract text from code files"""
        text = self._decode_bytes(content)
        lang = self.PROGRAMMING_LANGUAGES.get(ext, 'unknown')
        
        # Split by functions/classes for better chunking
        chunks = self._chunk_code(text, lang, chunk_size, chunk_overlap)
        
        return DocumentExtraction(
            text=text,
            chunks=chunks,
            word_count=len(text.split()),
            programming_language=lang,
            metadata={"language": lang, "lines": text.count('\n') + 1}
        )
    
    async def _extract_json(
        self,
        content: bytes,
        chunk_size: int,
        chunk_overlap: int
    ) -> DocumentExtraction:
        """Extract text from JSON"""
        text = self._decode_bytes(content)
        
        try:
            data = json.loads(text)
            # Pretty print for better readability
            formatted = json.dumps(data, indent=2)
            chunks = self._chunk_text(formatted, chunk_size, chunk_overlap)
            
            return DocumentExtraction(
                text=formatted,
                chunks=chunks,
                word_count=len(formatted.split()),
                metadata={"json_type": type(data).__name__}
            )
        except json.JSONDecodeError:
            # Fall back to plain text
            return await self._extract_text(content, chunk_size, chunk_overlap)
    
    async def _extract_yaml(
        self,
        content: bytes,
        chunk_size: int,
        chunk_overlap: int
    ) -> DocumentExtraction:
        """Extract text from YAML"""
        text = self._decode_bytes(content)
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        
        return DocumentExtraction(
            text=text,
            chunks=chunks,
            word_count=len(text.split()),
            metadata={"format": "yaml"}
        )
    
    async def _extract_csv(
        self,
        content: bytes,
        chunk_size: int,
        chunk_overlap: int
    ) -> DocumentExtraction:
        """Extract text from CSV"""
        text = self._decode_bytes(content)
        lines = text.strip().split('\n')
        
        # Get headers
        headers = lines[0] if lines else ""
        row_count = len(lines) - 1 if len(lines) > 1 else 0
        
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        
        return DocumentExtraction(
            text=text,
            chunks=chunks,
            word_count=len(text.split()),
            metadata={"headers": headers, "row_count": row_count, "format": "csv"}
        )
    
    async def _extract_text(
        self,
        content: bytes,
        chunk_size: int,
        chunk_overlap: int
    ) -> DocumentExtraction:
        """Extract plain text"""
        text = self._decode_bytes(content)
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        
        return DocumentExtraction(
            text=text,
            chunks=chunks,
            word_count=len(text.split())
        )
    
    def _decode_bytes(self, content: bytes) -> str:
        """Decode bytes to string with encoding detection"""
        if HAS_CHARDET:
            detected = chardet.detect(content)
            encoding = detected.get('encoding', 'utf-8') or 'utf-8'
        else:
            encoding = 'utf-8'
        
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            return content.decode('utf-8', errors='replace')
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[ExtractedChunk]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence or paragraph
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind('\n\n', start, end)
                if para_break > start + chunk_size // 2:
                    end = para_break
                else:
                    # Look for sentence break
                    for punct in ['. ', '! ', '? ', '\n']:
                        sent_break = text.rfind(punct, start, end)
                        if sent_break > start + chunk_size // 2:
                            end = sent_break + len(punct)
                            break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(ExtractedChunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end
                ))
                chunk_index += 1
            
            start = end - chunk_overlap
            if start < 0:
                start = 0
            if start >= len(text):
                break
        
        return chunks
    
    def _chunk_text_with_pages(
        self,
        pages: List[str],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[ExtractedChunk]:
        """Chunk text while preserving page information"""
        chunks = []
        chunk_index = 0
        current_pos = 0
        
        for page_num, page_text in enumerate(pages, 1):
            page_chunks = self._chunk_text(page_text, chunk_size, chunk_overlap)
            
            for chunk in page_chunks:
                chunk.page_number = page_num
                chunk.chunk_index = chunk_index
                chunk.start_char += current_pos
                chunk.end_char += current_pos
                chunks.append(chunk)
                chunk_index += 1
            
            current_pos += len(page_text) + 2  # +2 for \n\n separator
        
        return chunks
    
    def _chunk_markdown(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[ExtractedChunk]:
        """Chunk markdown by headers"""
        # Split by headers
        header_pattern = r'^(#{1,6}\s+.+)$'
        sections = re.split(header_pattern, text, flags=re.MULTILINE)
        
        chunks = []
        chunk_index = 0
        current_pos = 0
        current_header = ""
        
        for i, section in enumerate(sections):
            if re.match(header_pattern, section):
                current_header = section.strip()
            elif section.strip():
                # Add header context to content
                content = f"{current_header}\n\n{section.strip()}" if current_header else section.strip()
                
                if len(content) > chunk_size:
                    # Further chunk large sections
                    sub_chunks = self._chunk_text(content, chunk_size, chunk_overlap)
                    for sub in sub_chunks:
                        sub.chunk_index = chunk_index
                        sub.start_char += current_pos
                        sub.end_char += current_pos
                        sub.metadata = {"header": current_header}
                        chunks.append(sub)
                        chunk_index += 1
                else:
                    chunks.append(ExtractedChunk(
                        content=content,
                        chunk_index=chunk_index,
                        start_char=current_pos,
                        end_char=current_pos + len(section),
                        metadata={"header": current_header}
                    ))
                    chunk_index += 1
            
            current_pos += len(section)
        
        return chunks if chunks else self._chunk_text(text, chunk_size, chunk_overlap)
    
    def _chunk_code(
        self,
        text: str,
        language: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[ExtractedChunk]:
        """Chunk code by functions/classes"""
        # Patterns for different languages
        patterns = {
            'python': r'^((?:async\s+)?(?:def|class)\s+\w+)',
            'javascript': r'^((?:async\s+)?(?:function|class)\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:function|\())',
            'typescript': r'^((?:async\s+)?(?:function|class)\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:function|\())',
            'java': r'^(\s*(?:public|private|protected)?\s*(?:static)?\s*(?:class|interface|void|\w+)\s+\w+)',
            'go': r'^(func\s+(?:\([^)]+\)\s+)?\w+)',
            'rust': r'^((?:pub\s+)?(?:fn|struct|impl|enum|trait)\s+\w+)',
        }
        
        pattern = patterns.get(language)
        if not pattern:
            return self._chunk_text(text, chunk_size, chunk_overlap)
        
        # Split by function/class definitions
        parts = re.split(pattern, text, flags=re.MULTILINE)
        
        chunks = []
        chunk_index = 0
        current_pos = 0
        
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and re.match(pattern, parts[i], re.MULTILINE):
                # Combine definition with body
                content = parts[i] + parts[i + 1]
                i += 2
            else:
                content = parts[i]
                i += 1
            
            content = content.strip()
            if not content:
                continue
            
            if len(content) > chunk_size:
                sub_chunks = self._chunk_text(content, chunk_size, chunk_overlap)
                for sub in sub_chunks:
                    sub.chunk_index = chunk_index
                    chunks.append(sub)
                    chunk_index += 1
            else:
                chunks.append(ExtractedChunk(
                    content=content,
                    chunk_index=chunk_index,
                    start_char=current_pos,
                    end_char=current_pos + len(content)
                ))
                chunk_index += 1
            
            current_pos += len(content)
        
        return chunks if chunks else self._chunk_text(text, chunk_size, chunk_overlap)
    
    async def extract_media(
        self,
        content: bytes,
        filename: str,
        media_type: str,
        generate_description: bool = True,
        transcribe_audio: bool = True
    ) -> MediaExtraction:
        """
        Extract metadata and AI content from media files.
        """
        result = MediaExtraction()
        
        if media_type == 'image':
            result = await self._extract_image(content, generate_description)
        elif media_type in ('video', 'audio'):
            result = await self._extract_video_audio(
                content, filename, media_type, transcribe_audio
            )
        
        return result
    
    async def _extract_image(
        self,
        content: bytes,
        generate_description: bool
    ) -> MediaExtraction:
        """Extract metadata from image and optionally generate AI description"""
        result = MediaExtraction()
        
        if HAS_PIL:
            try:
                img = Image.open(io.BytesIO(content))
                result.width = img.width
                result.height = img.height
                result.format = img.format
            except Exception:
                pass
        
        # Generate AI description if OpenAI client available
        if generate_description and self.openai_client:
            try:
                import base64
                b64_image = base64.b64encode(content).decode('utf-8')
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this image in detail. Include what you see, any text visible, and the overall context. Be concise but thorough."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
                
                result.ai_description = response.choices[0].message.content
            except Exception as e:
                result.metadata = {"ai_error": str(e)}
        
        return result
    
    async def _extract_video_audio(
        self,
        content: bytes,
        filename: str,
        media_type: str,
        transcribe: bool
    ) -> MediaExtraction:
        """Extract metadata from video/audio and optionally transcribe"""
        result = MediaExtraction()
        
        # For now, just store basic info
        # Full video/audio processing would require moviepy and whisper
        result.metadata = {
            "note": "Video/audio transcription requires additional dependencies (moviepy, openai-whisper)"
        }
        
        return result
    
    async def generate_summary(
        self,
        text: str,
        max_length: int = 500
    ) -> Optional[str]:
        """Generate a summary of the document using AI"""
        if not self.openai_client:
            # Simple extractive summary
            sentences = text.split('. ')
            return '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else text[:max_length]
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes documents. Provide a concise summary."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize the following document in 2-3 sentences:\n\n{text[:4000]}"
                    }
                ],
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception:
            return None
    
    async def extract_key_topics(
        self,
        text: str,
        max_topics: int = 5
    ) -> List[str]:
        """Extract key topics from document using AI"""
        if not self.openai_client:
            # Simple keyword extraction
            words = re.findall(r'\b[A-Z][a-z]+\b', text)
            from collections import Counter
            return [w for w, _ in Counter(words).most_common(max_topics)]
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"Extract {max_topics} key topics from the document. Return as JSON array of strings."
                    },
                    {
                        "role": "user",
                        "content": text[:4000]
                    }
                ],
                max_tokens=100
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception:
            return []
    
    def save_file(
        self,
        content: bytes,
        filename: str,
        user_id: str
    ) -> str:
        """Save uploaded file to storage"""
        # Create user directory
        user_dir = os.path.join(self.storage_path, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Generate unique filename
        ext = os.path.splitext(filename)[1]
        unique_name = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(user_dir, unique_name)
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return file_path


# Singleton instance
_document_service: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """Get the document service instance"""
    global _document_service
    if _document_service is None:
        from app.services import get_embedding_service
        _document_service = DocumentService(
            embedding_service=get_embedding_service()
        )
    return _document_service
