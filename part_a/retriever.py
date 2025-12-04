"""
Vector store and retrieval system using ChromaDB.
Handles document indexing and similarity search.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from .text_processor import Chunk


logger = logging.getLogger(__name__)


class RetrievedChunk:
    """Represents a retrieved chunk with similarity score."""

    def __init__(self, content: str, metadata: Dict[str, Any], similarity_score: float):
        self.content = content
        self.metadata = metadata
        self.similarity_score = similarity_score


class VectorStoreRetriever:
    """Manages vector storage and retrieval using ChromaDB."""

    def __init__(self, embedding_model, persist_directory: Path, collection_name: str = "insurance_documents"):
        """
        Initialize the vector store retriever.

        Args:
            embedding_model: Model for generating embeddings (OpenAIEmbeddings or LocalEmbeddings)
            persist_directory: Directory to persist the database
            collection_name: Name of the collection in ChromaDB
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        logger.info(
            f"Initialized VectorStoreRetriever with collection '{collection_name}' "
            f"at {persist_directory}"
        )

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of text chunks to index
        """
        if not chunks:
            logger.warning("No chunks to add")
            return

        logger.info(f"Adding {len(chunks)} chunks to vector store")

        # Prepare data for ChromaDB
        ids = [f"chunk_{chunk.index}" for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.embed_texts(documents)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        logger.info(f"Successfully added {len(chunks)} chunks to vector store")

    def retrieve(
        self,
        query: str,
        k: int = 8,
        similarity_threshold: Optional[float] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: Search query
            k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score (optional)

        Returns:
            List of retrieved chunks with similarity scores
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Retrieving {k} chunks for query: '{query[:50]}...'")

        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)

        # Search in vector store
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        # Process results
        retrieved_chunks = []

        # ChromaDB returns nested lists even for single query
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, metadata, distance in zip(documents, metadatas, distances):
            # Convert distance to similarity (ChromaDB uses cosine distance)
            # Cosine distance = 1 - cosine similarity, so similarity = 1 - distance
            similarity_score = 1.0 - distance

            # Apply similarity threshold if specified
            if similarity_threshold is not None and similarity_score < similarity_threshold:
                logger.debug(f"Filtered out chunk with score {similarity_score:.3f}")
                continue

            retrieved_chunks.append(
                RetrievedChunk(
                    content=doc,
                    metadata=metadata,
                    similarity_score=similarity_score
                )
            )

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")

        return retrieved_chunks

    def count_documents(self) -> int:
        """Get the number of documents in the vector store."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        logger.warning("Clearing vector store")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Vector store cleared")
