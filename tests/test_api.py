"""Tests for FastAPI endpoints, request validation, and citation extraction."""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from backend.citations.formatter import extract_citations
from backend.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Test GET /api/health returns 200 and valid system telemetry."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "enterprise-rag-assistant"
    assert "models" in data
    assert "retrieval" in data


def test_chat_endpoint_valid_query():
    """Test POST /api/chat with a valid question."""
    mock_rag_response = {
        "answer": "The agreement requires 30 days written notice.",
        "citations": [
            {
                "document": "vendor_contract.pdf",
                "page": 3,
                "section": "Section 3: Term & Termination Notice Period",
            }
        ],
        "retrieval": {
            "candidate_count": 20,
            "final_context_count": 5,
        },
    }

    with patch("backend.api.chat.rag_service.answer_question", new_callable=AsyncMock) as mock_answer:
        mock_answer.return_value = mock_rag_response

        payload = {"question": "What is the termination notice period?"}
        response = client.post("/api/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The agreement requires 30 days written notice."
        assert len(data["citations"]) == 1
        assert data["citations"][0]["document"] == "vendor_contract.pdf"
        assert data["citations"][0]["page"] == 3
        assert data["retrieval"]["candidate_count"] == 20
        assert data["retrieval"]["final_context_count"] == 5


def test_chat_endpoint_empty_question():
    """Test POST /api/chat rejects empty questions with HTTP 422 Unprocessable Entity."""
    response = client.post("/api/chat", json={"question": ""})
    assert response.status_code == 422


def test_chat_endpoint_whitespace_only_question():
    """Test POST /api/chat rejects whitespace questions with HTTP 422."""
    response = client.post("/api/chat", json={"question": "   \n\t   "})
    assert response.status_code == 422


def test_chat_endpoint_missing_body():
    """Test POST /api/chat with missing body."""
    response = client.post("/api/chat", json={})
    assert response.status_code == 422


def test_extract_citations_deduplication():
    """Test extract_citations prevents duplicate entries and handles missing metadata safely."""
    docs = [
        Document(
            page_content="Text A",
            metadata={"source": "leave_policy.pdf", "page": 1, "section": "Annual Leave"},
        ),
        Document(
            page_content="Text B from same page and section",
            metadata={"source": "leave_policy.pdf", "page": 1, "section": "Annual Leave"},
        ),
        Document(
            page_content="Text C from another page",
            metadata={"source": "leave_policy.pdf", "page": 2, "section": "Sick Leave"},
        ),
        Document(
            page_content="Text D without page metadata",
            metadata={"source": "vendor_contract.pdf"},
        ),
    ]

    citations = extract_citations(docs)

    # 4 documents, but 1 duplicate -> 3 citations
    assert len(citations) == 3

    assert citations[0]["document"] == "leave_policy.pdf"
    assert citations[0]["page"] == 1
    assert citations[0]["section"] == "Annual Leave"

    assert citations[1]["document"] == "leave_policy.pdf"
    assert citations[1]["page"] == 2
    assert citations[1]["section"] == "Sick Leave"

    # Default fallback for missing page & section
    assert citations[2]["document"] == "vendor_contract.pdf"
    assert citations[2]["page"] == 1
    assert citations[2]["section"] == "General"
