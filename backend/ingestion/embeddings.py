"""Reusable embedding service for Enterprise RAG Assistant.
Uses HuggingFace BGE small embedding model (BAAI/bge-small-en-v1.5) via LangChain.
"""

import logging
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from backend.config import settings

logger = logging.getLogger(__name__)

_embedding_instance: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """Retrieve or initialize the centralized HuggingFace BGE embedding model singleton."""
    global _embedding_instance
    target_model = model_name or settings.EMBEDDING_MODEL

    if _embedding_instance is None:
        logger.info(f"Initializing HuggingFace BGE embedding model: {target_model}")
        _embedding_instance = HuggingFaceEmbeddings(
            model_name=target_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Embedding model initialized successfully: {target_model}")

    return _embedding_instance
