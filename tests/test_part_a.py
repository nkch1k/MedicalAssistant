"""
Unit tests for Part A - Console Application
Minimal but sufficient test coverage for core functionality
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import os

from part_a.document_loader import DocumentLoader
from part_a.text_processor import TextProcessor, Chunk
from part_a.embeddings import create_embedding_model
from part_a.rag_chain import RAGChain
from part_a.config import Config


class TestDocumentLoader:
    """Tests for document loading."""

    def test_clean_text(self):
        """Test text cleaning functionality."""
        loader = DocumentLoader()

        dirty_text = "  Line 1  \n\n\n  Line 2  \n\n\n\nLine 3  "
        cleaned = loader._clean_text(dirty_text)

        assert "Line 1" in cleaned
        assert "Line 2" in cleaned
        assert "Line 3" in cleaned
        assert "\n\n\n" not in cleaned

    def test_load_nonexistent_pdf(self):
        """Test error handling for non-existent PDF."""
        loader = DocumentLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_pdf(Path("nonexistent.pdf"))


class TestTextProcessor:
    """Tests for text processing and chunking."""

    def test_invalid_overlap(self):
        """Test validation of chunk parameters."""
        with pytest.raises(ValueError):
            TextProcessor(chunk_size=100, chunk_overlap=150)

    def test_process_simple_text(self):
        """Test basic text processing."""
        processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        text = "זהו טקסט בעברית.\n\nזהו פסקה שנייה."

        chunks = processor.process(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.content for chunk in chunks)


class TestEmbeddings:
    """Tests for embedding models."""

    def test_create_embedding_model_local(self):
        """Test creating local embedding model."""
        try:
            model = create_embedding_model(use_local=True)
            assert model is not None
            assert hasattr(model, 'embed_query')
            assert hasattr(model, 'embed_texts')
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_create_embedding_model_requires_api_key(self):
        """Test that OpenAI embeddings require API key."""
        with pytest.raises(ValueError):
            create_embedding_model(use_local=False, api_key=None)


class TestRAGChain:
    """Tests for RAG chain."""

    def test_rag_chain_initialization(self):
        """Test RAG chain initialization."""
        mock_retriever = Mock()
        mock_llm = Mock()

        chain = RAGChain(
            retriever=mock_retriever,
            llm=mock_llm,
            retrieval_k=3,
            similarity_threshold=0.7
        )

        assert chain.retrieval_k == 3
        assert chain.similarity_threshold == 0.7
        assert chain.retriever == mock_retriever
        assert chain.llm == mock_llm

    def test_empty_question(self):
        """Test validation of empty question."""
        mock_retriever = Mock()
        mock_llm = Mock()

        chain = RAGChain(mock_retriever, mock_llm)

        with pytest.raises(ValueError):
            chain.answer("")

    def test_no_chunks_retrieved(self):
        """Test handling when no relevant chunks found."""
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = []
        mock_llm = Mock()

        chain = RAGChain(mock_retriever, mock_llm)
        response = chain.answer("test question")

        assert "לא נמצא מידע רלוונטי" in response.answer
        assert not response.metadata["answer_generated"]


class TestConfig:
    """Tests for configuration."""

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key-12345",
        "CHUNK_SIZE": "600",
        "CHUNK_OVERLAP": "120",
        "RETRIEVAL_K": "4"
    })
    def test_config_initialization(self):
        """Test config initialization with environment variables."""
        config = Config()

        # Verify basic attributes exist and have expected defaults
        assert hasattr(config, 'chunk_size')
        assert hasattr(config, 'chunk_overlap')
        assert hasattr(config, 'retrieval_k')
        assert config.chunk_size == 600
        assert config.chunk_overlap == 120
        assert config.retrieval_k == 4
        assert config.openai_api_key == "test-key-12345"


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_text_processing(self):
        """Test complete text processing pipeline."""
        # Sample Hebrew text
        text = """
        סעיף 1: מידע על אקופונקטורה

        עלות הטיפול היא 8.22 ₪ לטיפול עבור 20 הטיפולים הראשונים.

        סעיף 2: תקופת אכשרה

        תקופת האכשרה היא 90 ימים מתחילת הפוליסה.
        """

        # Process text
        processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor.process(text, {"file_name": "test.pdf"})

        # Verify chunks were created
        assert len(chunks) > 0

        # Verify metadata
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "source_document" in chunk.metadata
            assert chunk.metadata["source_document"] == "test.pdf"

        # Verify content preservation
        full_content = " ".join(chunk.content for chunk in chunks)
        assert "8.22" in full_content or "8.22" in text
        assert "90" in full_content or "90" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
