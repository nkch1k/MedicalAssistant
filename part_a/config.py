"""
Configuration management for RAG system.
Handles environment variables and system settings.
"""

import os
from pathlib import Path


class Config:
    """Central configuration for the RAG system."""

    def __init__(self):
        # Load environment variables
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)

        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # File paths
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.vector_store_dir = self.project_root / "vector_store"

        # Document processing
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "600"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))

        # Retrieval settings
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "4"))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

        # Embeddings
        self.use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.local_embedding_model = os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

        # LLM settings
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "500"))

        # Validate
        if not self.use_local_embeddings and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when use_local_embeddings=False. "
                "Set it in your .env file or environment."
            )

        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    """Load configuration from environment."""
    return Config()
