"""Chat API route and request/response schemas for Enterprise RAG Assistant.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from backend.chains.rag_chain import rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Chat"])


class ChatRequest(BaseModel):
    """User question payload."""

    question: str = Field(
        ...,
        description="The question to query enterprise policies and contracts against.",
        examples=["What is the termination notice period?"],
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("Question cannot be empty or solely whitespace.")
        if len(clean) > 2000:
            raise ValueError("Question exceeds maximum length of 2000 characters.")
        return clean


class CitationItem(BaseModel):
    """Source citation item."""

    document: str = Field(..., description="Filename of source PDF")
    page: int = Field(..., description="1-indexed page number in the source PDF")
    section: str = Field(..., description="Document section header or topic")


class RetrievalStats(BaseModel):
    """Retrieval telemetry and diagnostics."""

    candidate_count: int = Field(..., description="Number of candidate chunks merged from hybrid retrieval")
    final_context_count: int = Field(..., description="Number of chunks reranked and provided to LLM context")


from backend.guardrails.schemas import GuardrailReport

class ChatResponse(BaseModel):
    """Chat answer with citations, retrieval telemetry, and AI guardrails status."""

    answer: str = Field(..., description="Grounded answer synthesized by the LLM")
    citations: List[CitationItem] = Field(default_factory=list, description="Preserved source citations")
    retrieval: RetrievalStats = Field(..., description="Retrieval counts")
    guardrails: Optional[GuardrailReport] = Field(default=None, description="AI Guardrails validation telemetry")


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Enterprise Policies & Contracts",
    description="Processes a user question through Hybrid Retrieval (Pinecone + BM25), BGE Reranking, and LangChain LLM.",
)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Chat endpoint processing questions through the enterprise RAG pipeline."""
    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    try:
        result = await rag_service.answer_question(clean_question)
        return ChatResponse(
            answer=result["answer"],
            citations=result["citations"],
            retrieval=RetrievalStats(
                candidate_count=result["retrieval"]["candidate_count"],
                final_context_count=result["retrieval"]["final_context_count"],
            ),
            guardrails=result.get("guardrails"),
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the request. Please verify server logs.",
        )
