"""
Main console application for RAG-based Q&A system.
Entry point for Part A.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .config import load_config
from .document_loader import DocumentLoader
from .text_processor import TextProcessor
from .embeddings import create_embedding_model
from .retriever import VectorStoreRetriever
from .rag_chain import RAGChain, LLMInterface


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("rag_service.log", encoding="utf-8")
    ]
)

logger = logging.getLogger(__name__)


class RAGApplication:
    """Main application for RAG Q&A system."""

    def __init__(self):
        """Initialize the RAG application."""
        self.config = load_config()
        self.rag_chain: Optional[RAGChain] = None

    def initialize_system(self, document_path: Path) -> None:
        """
        Initialize the RAG system with a document.

        Args:
            document_path: Path to the PDF document
        """
        logger.info("=" * 70)
        logger.info("Initializing RAG System")
        logger.info("=" * 70)

        try:
            # Step 1: Load document
            logger.info("Step 1/5: Loading document...")
            loader = DocumentLoader()
            text, metadata = loader.load_pdf(document_path)

            logger.info(f"✓ Document loaded: {metadata['total_pages']} pages, {metadata['total_characters']} characters")

            # Step 2: Process and chunk text
            logger.info("Step 2/5: Processing and chunking text...")
            processor = TextProcessor(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            chunks = processor.process(text, metadata)

            # Validate chunks
            validation_report = processor.validate_chunks(chunks)
            logger.info(f"✓ Created {validation_report['total_chunks']} chunks (avg size: {validation_report['avg_chunk_size']:.0f} chars)")

            # Step 3: Create embeddings
            logger.info("Step 3/5: Initializing embedding model...")
            embedding_model = create_embedding_model(
                use_local=self.config.use_local_embeddings,
                api_key=self.config.openai_api_key,
                model_name=self.config.embedding_model if not self.config.use_local_embeddings else self.config.local_embedding_model
            )
            logger.info(f"✓ Embedding model ready (dimension: {embedding_model.dimension})")

            # Step 4: Create vector store and index chunks
            logger.info("Step 4/5: Creating vector store and indexing chunks...")
            retriever = VectorStoreRetriever(
                embedding_model=embedding_model,
                persist_directory=self.config.vector_store_dir,
                collection_name="insurance_documents"
            )

            # Clear existing data (for fresh start)
            if retriever.count_documents() > 0:
                logger.info("Clearing existing vector store...")
                retriever.clear()

            retriever.add_chunks(chunks)
            logger.info(f"✓ Indexed {retriever.count_documents()} chunks in vector store")

            # Step 5: Initialize RAG chain
            logger.info("Step 5/5: Initializing RAG chain...")
            llm = LLMInterface(
                api_key=self.config.openai_api_key,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.max_tokens
            )

            self.rag_chain = RAGChain(
                retriever=retriever,
                llm=llm,
                retrieval_k=self.config.retrieval_k,
                similarity_threshold=self.config.similarity_threshold
            )

            logger.info("✓ RAG chain initialized")
            logger.info("=" * 70)
            logger.info("System ready!")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise

    def answer_question(self, question: str) -> str:
        """
        Answer a question using the RAG system.

        Args:
            question: User question

        Returns:
            Answer text
        """
        if not self.rag_chain:
            raise RuntimeError("System not initialized. Load a document first.")

        try:
            response = self.rag_chain.answer(question)
            return response.answer
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"שגיאה בעיבוד השאלה: {str(e)}"

    def run_interactive(self) -> None:
        """Run interactive console interface."""
        print("\n" + "=" * 70)
        print("מערכת שאלות ותשובות - רפואה משלימה")
        print("Insurance Q&A System - Complementary Medicine")
        print("=" * 70)
        print("\nפקודות זמינות / Available commands:")
        print("  - הקלד שאלה / Type a question")
        print("  - 'יציאה' / 'exit' / 'quit' - לסיום / to quit")
        print("=" * 70 + "\n")

        while True:
            try:
                # Get user input
                question = input("\n💬 שאלה / Question: ").strip()

                # Check for exit commands
                if question.lower() in ["יציאה", "exit", "quit", "q"]:
                    print("\nמסיים את התוכנית... / Shutting down...")
                    break

                # Skip empty input
                if not question:
                    continue

                # Process question
                print("\n🔍 מעבד שאלה... / Processing question...\n")
                answer = self.answer_question(question)

                # Display answer
                print(f"✅ תשובה / Answer:\n{answer}\n")
                print("-" * 70)

            except KeyboardInterrupt:
                print("\n\nמסיים את התוכנית... / Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in interactive loop: {e}")
                print(f"\n❌ שגיאה / Error: {e}\n")

        print("\nתודה שהשתמשת במערכת! / Thank you for using the system!")


def main():
    """Main entry point."""
    # Find PDF document
    config = load_config()
    pdf_files = list(config.data_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ Error: No PDF files found in {config.data_dir}")
        print(f"Please place your insurance document PDF in the 'data' directory.")
        return 1

    # Use the first PDF found
    document_path = pdf_files[0]
    print(f"📄 Using document: {document_path.name}\n")

    try:
        # Initialize application
        app = RAGApplication()
        app.initialize_system(document_path)

        # Run interactive interface
        app.run_interactive()

        return 0

    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
