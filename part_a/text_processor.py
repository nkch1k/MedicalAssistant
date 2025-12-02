"""
Text processing and chunking strategies.
Optimized for Hebrew text with semantic preservation.
"""

import logging
from typing import List, Dict, Any
import re


logger = logging.getLogger(__name__)


class Chunk:
    """Represents a text chunk with metadata."""

    def __init__(self, content: str, index: int, metadata: Dict[str, Any]):
        self.content = content
        self.index = index
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.content)


class TextProcessor:
    """Processes and chunks text for RAG retrieval."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        """
        Initialize text processor.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._section_pattern = re.compile(r"סעיף\s+\d+", re.UNICODE)

    def process(self, text: str, document_metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Process and chunk text with a single unified algorithm.

        Args:
            text: Full document text
            document_metadata: Metadata about the source document

        Returns:
            List of text chunks with metadata
        """
        if not text or not text.strip():
            raise ValueError("Cannot process empty text")

        logger.info(f"Processing text of {len(text)} characters")

        # Split by paragraphs
        paragraphs = re.split(r"\n\n+", text)
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If paragraph is too large, split it by sentences
            if len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                chunks.extend(self._split_large_paragraph(paragraph))
                continue

            # If adding paragraph exceeds chunk size, start new chunk
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap to all chunks
        chunks = self._apply_overlap(chunks)

        # Create Chunk objects with metadata
        chunk_objects = []
        for idx, chunk_text in enumerate(chunks):
            metadata = {
                "chunk_index": idx,
                "chunk_size": len(chunk_text),
                "source_document": document_metadata.get("file_name") if document_metadata else None,
            }

            # Try to identify which section this chunk belongs to
            section_match = self._section_pattern.search(chunk_text)
            if section_match:
                metadata["section"] = section_match.group()

            chunk_objects.append(Chunk(content=chunk_text, index=idx, metadata=metadata))

        logger.info(f"Created {len(chunk_objects)} chunks")
        return chunk_objects

    def _split_large_paragraph(self, text: str) -> List[str]:
        """
        Split a large paragraph by sentences.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # Split by common sentence endings
        sentences = re.split(r'([.!?]\s+)', text)

        # Rejoin sentences with their punctuation
        reconstructed = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                reconstructed.append(sentences[i] + sentences[i + 1])
            else:
                reconstructed.append(sentences[i])
        if len(sentences) % 2 == 1:
            reconstructed.append(sentences[-1])

        chunks = []
        current_chunk = ""

        for sentence in reconstructed:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If single sentence is too large, split by character count
            if len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                chunks.extend(self._split_by_size(sentence))
                continue

            # If adding sentence exceeds chunk size, start new chunk
            if current_chunk and len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_by_size(self, text: str) -> List[str]:
        """
        Split text by character count (last resort for very long text without delimiters).

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between consecutive chunks.

        Args:
            chunks: List of chunks without overlap

        Returns:
            List of chunks with overlap
        """
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks

        overlapped_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            # Take overlap from previous chunk
            prev_chunk = chunks[i - 1]
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk

            # Prepend overlap to current chunk
            overlapped_chunk = overlap_text + " " + chunks[i]
            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks

    def validate_chunks(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Validate chunk quality with simple checks."""
        if not chunks:
            return {"valid": False, "error": "No chunks created"}

        avg_size = sum(len(chunk) for chunk in chunks) / len(chunks)

        return {
            "valid": True,
            "total_chunks": len(chunks),
            "avg_chunk_size": avg_size
        }
