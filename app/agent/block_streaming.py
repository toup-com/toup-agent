"""
Block Streaming — Stream tool calls and text blocks incrementally.

Instead of waiting for the full agent response, stream individual
blocks (text chunks, tool calls, tool results) as they arrive.
Each block has a type and can be rendered incrementally in the UI.

Usage:
    from app.agent.block_streaming import BlockStream, BlockType

    stream = BlockStream(session_id="s1")
    stream.add_text("Hello ")
    stream.add_text("world!")
    stream.add_tool_call("web_search", {"query": "weather"})
    stream.add_tool_result("web_search", {"results": [...]})

    for block in stream.blocks:
        print(block.type, block.content)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class BlockType(str, Enum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    ERROR = "error"
    STATUS = "status"
    MEDIA = "media"


@dataclass
class StreamBlock:
    """A single block in the response stream."""
    type: BlockType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    index: int = 0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "index": self.index,
        }


class BlockStream:
    """
    Manages a stream of response blocks for incremental rendering.

    Blocks are added as the agent produces them. Listeners are notified
    of each new block for real-time streaming to the UI.
    """

    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        self._blocks: List[StreamBlock] = []
        self._listeners: List[Callable[[StreamBlock], None]] = []
        self._finalized: bool = False
        self._created_at: float = time.time()
        self._text_buffer: str = ""

    @property
    def blocks(self) -> List[StreamBlock]:
        return list(self._blocks)

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    def on_block(self, callback: Callable[[StreamBlock], None]) -> None:
        """Register a listener for new blocks."""
        self._listeners.append(callback)

    def _emit(self, block: StreamBlock) -> None:
        """Notify all listeners of a new block."""
        for listener in self._listeners:
            try:
                listener(block)
            except Exception as e:
                logger.error(f"[STREAM] Listener error: {e}")

    def _add_block(self, block_type: BlockType, content: Any, metadata: Optional[Dict] = None) -> StreamBlock:
        """Add a block and notify listeners."""
        if self._finalized:
            raise RuntimeError("Stream is finalized, cannot add blocks")

        block = StreamBlock(
            type=block_type,
            content=content,
            metadata=metadata or {},
            index=len(self._blocks),
        )
        self._blocks.append(block)
        self._emit(block)
        return block

    def add_text(self, text: str) -> StreamBlock:
        """Add a text chunk to the stream."""
        self._text_buffer += text
        return self._add_block(BlockType.TEXT, text)

    def add_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> StreamBlock:
        """Add a tool call block."""
        return self._add_block(
            BlockType.TOOL_CALL,
            {"tool": tool_name, "arguments": arguments},
        )

    def add_tool_result(self, tool_name: str, result: Any, *, success: bool = True) -> StreamBlock:
        """Add a tool result block."""
        return self._add_block(
            BlockType.TOOL_RESULT,
            {"tool": tool_name, "result": result, "success": success},
        )

    def add_thinking(self, thought: str) -> StreamBlock:
        """Add a thinking block (for extended thinking mode)."""
        return self._add_block(BlockType.THINKING, thought)

    def add_error(self, error: str, *, tool_name: Optional[str] = None) -> StreamBlock:
        """Add an error block."""
        return self._add_block(
            BlockType.ERROR,
            error,
            metadata={"tool": tool_name} if tool_name else {},
        )

    def add_status(self, status: str) -> StreamBlock:
        """Add a status update block."""
        return self._add_block(BlockType.STATUS, status)

    def add_media(self, media_type: str, url: str, **kwargs) -> StreamBlock:
        """Add a media block (image, audio, video)."""
        return self._add_block(
            BlockType.MEDIA,
            {"media_type": media_type, "url": url, **kwargs},
        )

    def finalize(self) -> None:
        """Mark the stream as complete."""
        self._finalized = True

    def get_full_text(self) -> str:
        """Get the concatenated text from all text blocks."""
        return self._text_buffer

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Get all tool call blocks."""
        return [
            b.content for b in self._blocks
            if b.type == BlockType.TOOL_CALL
        ]

    def get_tool_results(self) -> List[Dict[str, Any]]:
        """Get all tool result blocks."""
        return [
            b.content for b in self._blocks
            if b.type == BlockType.TOOL_RESULT
        ]

    def get_errors(self) -> List[str]:
        """Get all error blocks."""
        return [
            b.content for b in self._blocks
            if b.type == BlockType.ERROR
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full stream."""
        return {
            "session_id": self.session_id,
            "blocks": [b.to_dict() for b in self._blocks],
            "block_count": len(self._blocks),
            "full_text": self.get_full_text(),
            "finalized": self._finalized,
        }

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the stream content."""
        type_counts: Dict[str, int] = {}
        for b in self._blocks:
            type_counts[b.type.value] = type_counts.get(b.type.value, 0) + 1

        return {
            "session_id": self.session_id,
            "total_blocks": len(self._blocks),
            "by_type": type_counts,
            "text_length": len(self._text_buffer),
            "finalized": self._finalized,
            "duration_seconds": round(time.time() - self._created_at, 2),
        }


class BlockStreamRegistry:
    """Registry to manage multiple active block streams."""

    def __init__(self):
        self._streams: Dict[str, BlockStream] = {}

    def create(self, session_id: str) -> BlockStream:
        """Create a new block stream for a session."""
        stream = BlockStream(session_id=session_id)
        self._streams[session_id] = stream
        return stream

    def get(self, session_id: str) -> Optional[BlockStream]:
        """Get an active stream by session ID."""
        return self._streams.get(session_id)

    def remove(self, session_id: str) -> bool:
        """Remove a stream."""
        return self._streams.pop(session_id, None) is not None

    def list_active(self) -> List[str]:
        """List session IDs with active (non-finalized) streams."""
        return [
            sid for sid, s in self._streams.items()
            if not s.is_finalized
        ]

    @property
    def count(self) -> int:
        return len(self._streams)


# ── Singleton ────────────────────────────────────────────
_registry: Optional[BlockStreamRegistry] = None


def get_block_stream_registry() -> BlockStreamRegistry:
    """Get the global block stream registry."""
    global _registry
    if _registry is None:
        _registry = BlockStreamRegistry()
    return _registry
