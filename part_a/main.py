"""
Main console application for RAG-based Q&A system.
Entry point for Part A.
"""

import logging
import sys
import argparse
from pathlib import Path
from typing import Optional, List

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

    def initialize_system(self, document_path: Path) -> tuple[int, int]:
        """
        Initialize the RAG system with a document.

        Args:
            document_path: Path to the PDF document

        Returns:
            Tuple of (total_pages, total_chunks)
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

            # Return actual metadata
            return metadata['total_pages'], validation_report['total_chunks']

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
        print("Insurance Q&A System - Complementary Medicine")
        print("=" * 70)
        print("\nAvailable commands:")
        print("  - Type your question (Hebrew supported)")
        print("  - 'exit' / 'quit' - to quit")
        print("=" * 70 + "\n")

        while True:
            try:
                # Get user input
                question = input("\nQuestion: ").strip()

                # Check for exit commands
                if question.lower() in ["exit", "quit", "q"]:
                    print("\nShutting down...")
                    break

                # Skip empty input
                if not question:
                    continue

                # Process question
                print("\nProcessing question...\n")
                answer = self.answer_question(question)

                # Display answer
                print(f"Answer:\n{answer}\n")
                print("-" * 70)

            except KeyboardInterrupt:
                print("\n\nShutting down...")
                break
            except Exception as e:
                logger.error(f"Error in interactive loop: {e}")
                print(f"\nError: {e}\n")

        print("\nThank you for using the system!")

    def run_from_file(self, questions_file: Path) -> None:
        """
        Run Q&A from questions file (for RTL terminal issues).

        Args:
            questions_file: Path to text file with questions (one per line)
        """
        if not questions_file.exists():
            print(f"Error: Questions file not found: {questions_file}")
            return

        print("\n" + "=" * 70)
        print("Insurance Q&A System - Batch Mode")
        print("=" * 70)
        print(f"Reading questions from: {questions_file.name}\n")

        # Read questions
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]

        if not questions:
            print("Error: No questions found in file")
            return

        print(f"Found {len(questions)} question(s)\n")
        print("=" * 70 + "\n")

        # Process each question
        for i, question in enumerate(questions, 1):
            print(f"Question {i}/{len(questions)}:")
            print(f"{question}\n")
            print("Processing...\n")

            answer = self.answer_question(question)

            print(f"Answer:\n{answer}\n")
            print("-" * 70 + "\n")

        print("=" * 70)
        print("Batch processing complete!")
        print("=" * 70)


def main():
    """Main entry point."""
    # Configure stdout encoding for Windows console
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="RAG-based Insurance Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default):
  python -m part_a.main

  # Batch mode from questions file:
  python -m part_a.main --file test_questions.txt
        """
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Path to text file with questions (one per line) for batch processing'
    )

    args = parser.parse_args()

    # Find PDF document
    config = load_config()
    pdf_files = list(config.data_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"Error: No PDF files found in {config.data_dir}")
        print(f"Please place your insurance document PDF in the 'data' directory.")
        return 1

    # Use the first PDF found
    document_path = pdf_files[0]
    print(f"Using document: {document_path.name}\n")

    try:
        # Initialize application
        app = RAGApplication()
        pages, chunks = app.initialize_system(document_path)

        # Run in batch or interactive mode
        if args.file:
            questions_file = Path(args.file)
            app.run_from_file(questions_file)
        else:
            app.run_interactive()

        return 0

    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\nFatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
