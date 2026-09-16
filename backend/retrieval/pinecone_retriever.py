"""Pinecone semantic vector retriever module.
Queries the indexed Pinecone vector database using BGE embeddings.
"""

import logging
from typing import List, Optional
from langchain_core.documents import Document
from backend.config import settings
from backend.ingestion.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class PineconeRetriever:
    """Retriever for performing semantic vector search over Pinecone."""

    def __init__(self, index_name: Optional[str] = None, namespace: Optional[str] = None):
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.namespace = namespace or settings.PINECONE_NAMESPACE
        self._vector_store = None

    def _get_vector_store(self):
        """Lazy initialization of the PineconeVectorStore."""
        if self._vector_store is None:
            if not settings.PINECONE_API_KEY or settings.PINECONE_API_KEY.startswith("your_"):
                logger.warning("PINECONE_API_KEY is not configured; vector search will return empty results.")
                return None

            try:
                from langchain_pinecone import PineconeVectorStore

                embeddings = get_embedding_model()
                self._vector_store = PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=embeddings,
                    namespace=self.namespace,
                    pinecone_api_key=settings.PINECONE_API_KEY,
                )
            except Exception as e:
                logger.error(f"Failed to initialize PineconeVectorStore: {e}")
                return None

        return self._vector_store

    def search(self, query: str, top_k: int = settings.VECTOR_TOP_K) -> List[Document]:
        """Perform semantic similarity search on Pinecone index.
        Returns top_k matching Document objects with metadata preserved.
        """
        store = self._get_vector_store()
        if store is None:
            logger.debug("Pinecone store is unavailable; returning empty vector results.")
            return []

        try:
            results = store.similarity_search(query, k=top_k)
            logger.info(f"Pinecone retrieved {len(results)} candidate chunks for query: '{query[:40]}...'")
            return results
        except Exception as e:
            logger.error(f"Pinecone search error: {e}", exc_info=True)
            return []
