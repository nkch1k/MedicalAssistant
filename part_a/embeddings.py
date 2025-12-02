"""
Embeddings generation for document chunks.
Supports both OpenAI and local multilingual models.
"""

import logging
from typing import List, Optional


logger = logging.getLogger(__name__)


class OpenAIEmbeddings:
    """OpenAI embeddings implementation."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embeddings.

        Args:
            api_key: OpenAI API key
            model: Model name
        """
        self.api_key = api_key
        self.model = model
        self.dimension = 1536 if "text-embedding-3-small" in model else 3072

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        logger.info(f"Initialized OpenAI embeddings with model: {model}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )

            embeddings = [item.embedding for item in response.data]
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        if not query:
            raise ValueError("Query cannot be empty")

        embeddings = self.embed_texts([query])
        return embeddings[0]


class LocalEmbeddings:
    """Local multilingual embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """
        Initialize local embeddings.

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(f"Initialized local embeddings with model: {model_name}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            embeddings_list = embeddings.tolist()
            logger.debug(f"Generated {len(embeddings_list)} embeddings")
            return embeddings_list

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        if not query:
            raise ValueError("Query cannot be empty")

        embeddings = self.embed_texts([query])
        return embeddings[0]


def create_embedding_model(
    use_local: bool = False,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
):
    """
    Create an embedding model.

    Args:
        use_local: Whether to use local embeddings
        api_key: OpenAI API key (required if use_local=False)
        model_name: Model name (optional)

    Returns:
        OpenAIEmbeddings or LocalEmbeddings instance
    """
    if use_local:
        model_name = model_name or "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        return LocalEmbeddings(model_name=model_name)
    else:
        if not api_key:
            raise ValueError("API key required for OpenAI embeddings")
        model_name = model_name or "text-embedding-3-small"
        return OpenAIEmbeddings(api_key=api_key, model=model_name)
