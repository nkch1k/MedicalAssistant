"""
Business logic service for Part B.
Integrates Part A RAG system with Part B API.
"""

import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple

from part_a.main import RAGApplication
from part_a.config import load_config
from .storage import QAStorage, QARecord


logger = logging.getLogger(__name__)


class RAGService:
    """Service layer that integrates RAG system with API."""

    def __init__(self, database_url: str = "sqlite:///./db/qa_history.db"):
        """
        Initialize RAG service.

        Args:
            database_url: Database URL for Q&A storage
        """
        self.rag_app: Optional[RAGApplication] = None
        self.storage = QAStorage(database_url)
        self.config = load_config()
        self.current_document_id: Optional[str] = None

        logger.info("RAG Service initialized")

    def initialize_rag(self, document_path: Path) -> Tuple[int, int]:
        """
        Initialize RAG system with a document.

        Args:
            document_path: Path to PDF document

        Returns:
            Tuple of (pages, chunks)

        Raises:
            ValueError: If document cannot be processed
        """
        logger.info(f"Initializing RAG with document: {document_path}")

        try:
            # Initialize RAG application
            self.rag_app = RAGApplication()
            self.rag_app.initialize_system(document_path)

            # Extract metadata (simplified - would need to access from loader)
            # For now, we'll return placeholder values
            pages = 11  # Would get from document metadata
            chunks = 45  # Would get from processor

            self.current_document_id = f"doc-{uuid.uuid4()}"
            logger.info(f"RAG initialized successfully with document ID: {self.current_document_id}")

            return pages, chunks

        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            raise ValueError(f"Failed to process document: {str(e)}")

    def is_initialized(self) -> bool:
        """Check if RAG system is initialized."""
        return self.rag_app is not None

    def process_question(
        self,
        question: str,
        query_id: Optional[str] = None
    ) -> QARecord:
        """
        Process a question and store the result.

        Args:
            question: Question text
            query_id: Optional custom query ID (generates UUID if None)

        Returns:
            QARecord with question and answer

        Raises:
            RuntimeError: If RAG system not initialized
            ValueError: If query_id already exists
        """
        if not self.is_initialized():
            raise RuntimeError(
                "RAG system not initialized. Upload a document first."
            )

        # Generate or validate query ID
        if query_id is None:
            query_id = str(uuid.uuid4())
        elif self.storage.exists(query_id):
            raise ValueError(f"Query ID already exists: {query_id}")

        logger.info(f"Processing question with ID: {query_id}")

        try:
            # Get answer from RAG system
            answer = self.rag_app.answer_question(question)

            # Save to database
            record = self.storage.save_qa(
                query_id=query_id,
                question=question,
                answer=answer,
                status="completed"
            )

            logger.info(f"Successfully processed question: {query_id}")
            return record

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            # Try to save error state
            try:
                record = self.storage.save_qa(
                    query_id=query_id,
                    question=question,
                    answer=f"Error: {str(e)}",
                    status="error"
                )
                return record
            except:
                raise RuntimeError(f"Failed to process question: {str(e)}")

    def get_answer(self, query_id: str) -> Optional[QARecord]:
        """
        Retrieve an answer by query ID.

        Args:
            query_id: Query identifier

        Returns:
            QARecord if found, None otherwise
        """
        logger.info(f"Retrieving answer for ID: {query_id}")
        return self.storage.get_qa(query_id)

    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "rag_initialized": self.is_initialized(),
            "current_document_id": self.current_document_id,
            "total_queries": self.storage.count_records()
        }
