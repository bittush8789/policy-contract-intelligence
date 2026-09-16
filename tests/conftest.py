"""Pytest fixtures and configuration for Enterprise RAG Assistant test suite.
Ensures tests run completely offline without requiring live API keys or external services.
"""

import os
import sys
from pathlib import Path
import pytest
from typing import List
from langchain_core.documents import Document

# Ensure root workspace is on python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Enforce offline test flags
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["PINECONE_API_KEY"] = "mock_pinecone_key"
os.environ["GROQ_API_KEY"] = "mock_groq_key"


@pytest.fixture
def sample_documents() -> List[Document]:
    """Provide a deterministic fixture of enterprise document chunks."""
    return [
        Document(
            page_content="All full-time permanent employees are entitled to 25 days of paid annual leave per year.",
            metadata={
                "source": "leave_policy.pdf",
                "page": 1,
                "section": "Section 1: Annual Leave Entitlement",
                "document_type": "policy",
                "chunk_id": "leave_policy_1_01",
            },
        ),
        Document(
            page_content="Employees receive up to 10 days of paid sick leave per year for illness or doctor appointments.",
            metadata={
                "source": "leave_policy.pdf",
                "page": 2,
                "section": "Section 2: Sick & Medical Leave",
                "document_type": "policy",
                "chunk_id": "leave_policy_2_01",
            },
        ),
        Document(
            page_content="Either party may terminate this agreement without cause upon thirty (30) days written notice.",
            metadata={
                "source": "vendor_contract.pdf",
                "page": 3,
                "section": "Section 3: Term & Termination Notice Period",
                "document_type": "contract",
                "chunk_id": "vendor_contract_3_01",
            },
        ),
        Document(
            page_content="Purchases exceeding $50,000 require Vice President and Chief Financial Officer approval with three bids.",
            metadata={
                "source": "procurement_policy.pdf",
                "page": 2,
                "section": "Section 2: Approval Thresholds",
                "document_type": "policy",
                "chunk_id": "procurement_policy_2_01",
            },
        ),
        Document(
            page_content="All corporate accounts require passwords with a minimum length of 14 characters and Multi-Factor Authentication.",
            metadata={
                "source": "information_security_policy.pdf",
                "page": 1,
                "section": "Section 1: Password & Authentication Standards",
                "document_type": "policy",
                "chunk_id": "information_security_policy_1_01",
            },
        ),
    ]
