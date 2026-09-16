"""BGE Reranker module using BAAI/bge-reranker-v2-m3 via sentence-transformers CrossEncoder.
Reranks candidate chunks based on true cross-attention between the query and each chunk.
"""

import logging
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from backend.config import settings

logger = logging.getLogger(__name__)

_reranker_instance = None


class BGEReranker:
    """BGE CrossEncoder Reranker for enterprise document chunks."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None

    def _get_model(self):
        """Lazy-load the CrossEncoder model with fallback."""
        global _reranker_instance
        if self._model is None:
            if _reranker_instance is not None:
                self._model = _reranker_instance
            else:
                try:
                    logger.info(f"Loading CrossEncoder reranker model: {self.model_name}...")
                    from sentence_transformers import CrossEncoder

                    # Load model on CPU / available device
                    self._model = CrossEncoder(self.model_name)
                    _reranker_instance = self._model
                    logger.info("CrossEncoder reranker loaded successfully.")
                except Exception as e:
                    logger.warning(
                        f"Could not load CrossEncoder model '{self.model_name}' ({e}). "
                        "Falling back to hybrid ranking order."
                    )
                    self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = settings.RERANK_TOP_K,
    ) -> List[Document]:
        """Rerank candidate documents against the query.
        Returns top_k documents sorted descending by relevance score,
        with metadata fully preserved and rerank_score attached.
        """
        if not documents:
            logger.warning("Reranker received empty document list.")
            return []

        if len(documents) <= 1:
            return documents[:top_k]

        model = self._get_model()
        if model is None:
            # Fallback: preserve incoming hybrid order and assign simulated scores
            fallback_docs = []
            for rank, doc in enumerate(documents[:top_k], start=1):
                new_meta = dict(doc.metadata)
                new_meta["rerank_score"] = round(1.0 / rank, 4)
                fallback_docs.append(Document(page_content=doc.page_content, metadata=new_meta))
            return fallback_docs

        # Build cross-encoder input pairs: (query, text)
        pairs = [(query, doc.page_content) for doc in documents]

        try:
            scores = model.predict(pairs)
        except Exception as e:
            logger.error(f"Reranker scoring failed: {e}", exc_info=True)
            # Safe fallback: return candidates up to top_k in original order
            return documents[:top_k]

        # Combine docs, scores, and preserve metadata
        scored_docs: List[tuple[float, Document]] = []
        for score, doc in zip(scores, documents):
            score_val = float(score)
            new_meta = dict(doc.metadata)
            new_meta["rerank_score"] = round(score_val, 6)
            scored_doc = Document(
                page_content=doc.page_content,
                metadata=new_meta,
            )
            scored_docs.append((score_val, scored_doc))

        # Sort descending by cross-encoder score
        scored_docs.sort(key=lambda item: item[0], reverse=True)

        top_reranked = [doc for _, doc in scored_docs[:top_k]]

        logger.info(
            f"Reranked {len(documents)} candidates to top {len(top_reranked)} chunks. "
            f"Top score: {top_reranked[0].metadata.get('rerank_score')}"
        )
        return top_reranked
