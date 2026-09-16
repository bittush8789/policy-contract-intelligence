"""Tests for BGE CrossEncoder reranker."""

from unittest.mock import MagicMock
from langchain_core.documents import Document
from backend.retrieval.reranker import BGEReranker


def test_reranker_scoring_and_sorting(sample_documents):
    """Test that BGEReranker uses CrossEncoder predictions to score and re-order chunks."""
    reranker = BGEReranker()

    # Mock the internal CrossEncoder model
    mock_model = MagicMock()
    # Let's say model assigns higher scores to doc 2 (termination) and doc 0 (annual leave)
    # sample_documents has 5 docs; mock predict returns 5 scores
    mock_model.predict.return_value = [0.45, 0.12, 0.98, 0.23, 0.77]
    reranker._model = mock_model

    query = "What is the termination notice period?"
    reranked = reranker.rerank(query, sample_documents, top_k=3)

    assert len(reranked) == 3

    # The doc with score 0.98 (vendor_contract.pdf) should be first
    assert reranked[0].metadata["source"] == "vendor_contract.pdf"
    assert reranked[0].metadata["rerank_score"] == 0.98

    # The doc with score 0.77 (information_security_policy.pdf) should be second
    assert reranked[1].metadata["source"] == "information_security_policy.pdf"
    assert reranked[1].metadata["rerank_score"] == 0.77

    # The doc with score 0.45 (leave_policy.pdf page 1) should be third
    assert reranked[2].metadata["source"] == "leave_policy.pdf"
    assert reranked[2].metadata["rerank_score"] == 0.45

    # Verify original metadata is preserved
    for doc in reranked:
        assert "source" in doc.metadata
        assert "page" in doc.metadata
        assert "section" in doc.metadata
        assert "chunk_id" in doc.metadata


def test_reranker_empty_or_single_document():
    """Test reranker edge cases: empty list and single document."""
    reranker = BGEReranker()
    assert reranker.rerank("any query", []) == []

    single_doc = [Document(page_content="Single document", metadata={"source": "single.pdf"})]
    res = reranker.rerank("any query", single_doc)
    assert len(res) == 1
    assert res[0].metadata["source"] == "single.pdf"


def test_reranker_fallback_when_model_is_none(sample_documents):
    """Test reranker graceful fallback when model cannot be loaded (offline/fallback mode)."""
    reranker = BGEReranker()
    reranker._model = None
    reranker._get_model = MagicMock(return_value=None)

    reranked = reranker.rerank("query", sample_documents, top_k=3)
    assert len(reranked) == 3
    assert "rerank_score" in reranked[0].metadata
