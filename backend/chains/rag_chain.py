"""Orchestration service for the complete Enterprise RAG pipeline:
Hybrid Retrieval (Pinecone + BM25) -> Reranking (CrossEncoder) -> Context Building -> Prompt -> LLM.
"""

import logging
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langsmith import traceable
from backend.chains.prompts import RAG_PROMPT_TEMPLATE
from backend.citations.formatter import extract_citations
from backend.config import settings
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.reranker import BGEReranker

logger = logging.getLogger(__name__)

_llm_instance: Optional[BaseChatModel] = None


def get_llm() -> BaseChatModel:
    """Retrieve or initialize the centralized ChatGroq LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        logger.info(f"Initializing ChatGroq model: {settings.GROQ_MODEL}")
        from langchain_groq import ChatGroq

        _llm_instance = ChatGroq(
            model=settings.GROQ_MODEL,
            temperature=0.0,
            api_key=settings.GROQ_API_KEY or "dummy_key_for_offline",
        )
    return _llm_instance


class RAGService:
    """Enterprise RAG Service orchestrating hybrid retrieval, reranking, context formatting, and LLM inference."""

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[BGEReranker] = None,
        llm: Optional[BaseChatModel] = None,
    ):
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.reranker = reranker or BGEReranker()
        self._llm = llm

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def build_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into structured context blocks for the prompt."""
        if not documents:
            return "No relevant documents found."

        context_blocks = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", "N/A")
            section = doc.metadata.get("section", "General")

            header = f"[EXCERPT {i} | Document: {source} | Page: {page} | Section: {section}]"
            content = doc.page_content.strip()
            context_blocks.append(f"{header}\n{content}")

        return "\n\n" + "\n\n---\n\n".join(context_blocks) + "\n\n"

    @traceable(
        run_type="chain",
        name="RAG Pipeline",
        tags=["rag", "enterprise", "hybrid-retrieval"],
    )
    async def answer_question(self, question: str) -> Dict[str, Any]:
        """Execute end-to-end RAG workflow for a user query."""
        clean_question = question.strip()
        if not clean_question:
            return {
                "answer": "Please provide a valid question.",
                "citations": [],
                "retrieval": {"candidate_count": 0, "final_context_count": 0},
            }

        logger.info(f"Processing question: '{clean_question}'")

        # 0. Pre-retrieval Input Guardrails
        from backend.guardrails.service import guardrails_service

        input_eval = guardrails_service.evaluate_input(clean_question)
        if not input_eval.allowed:
            logger.warning(f"Question blocked by Input Guardrails: {input_eval.blocked_reason}")
            report = guardrails_service.create_report(input_eval)
            return {
                "answer": input_eval.blocked_reason or "Request blocked by enterprise security guardrails.",
                "citations": [],
                "retrieval": {"candidate_count": 0, "final_context_count": 0},
                "guardrails": report.model_dump(),
            }

        effective_query = input_eval.sanitized_query

        # 1. Hybrid Retrieval (Pinecone + BM25) -> Top 20-30 Candidates
        candidate_chunks = self.hybrid_retriever.retrieve(
            query=effective_query,
            vector_top_k=settings.VECTOR_TOP_K,
            bm25_top_k=settings.BM25_TOP_K,
            hybrid_top_k=settings.HYBRID_TOP_K,
        )

        candidate_count = len(candidate_chunks)

        # Handle zero retrieval edge case
        if candidate_count == 0:
            logger.warning("Zero candidate chunks returned from hybrid retrieval.")
            safe_ans = "I could not find enough information in the available documents to answer this question."
            output_eval = guardrails_service.evaluate_output(safe_ans, context_present=False)
            report = guardrails_service.create_report(input_eval, output_eval)
            return {
                "answer": output_eval.sanitized_answer,
                "citations": [],
                "retrieval": {"candidate_count": 0, "final_context_count": 0},
                "guardrails": report.model_dump(),
            }

        # 2. BGE CrossEncoder Reranking -> Top 5 Context Chunks
        top_context_chunks = self.reranker.rerank(
            query=effective_query,
            documents=candidate_chunks,
            top_k=settings.RERANK_TOP_K,
        )

        final_context_count = len(top_context_chunks)

        # 3. Context Construction
        formatted_context = self.build_context(top_context_chunks)

        # 4. Prompt Assembly & LLM Generation
        prompt_messages = RAG_PROMPT_TEMPLATE.format_messages(
            context=formatted_context,
            question=effective_query,
        )

        try:
            # Check if Groq LLM API key is configured
            if not settings.GROQ_API_KEY:
                # Provide an informative simulated response for development when no key is set yet
                logger.warning("No GROQ_API_KEY configured. Returning contextual simulation.")
                top_doc = top_context_chunks[0]
                raw_answer = (
                    f"Based on {top_doc.metadata.get('source')}, Page {top_doc.metadata.get('page')}, "
                    f"Section '{top_doc.metadata.get('section')}':\n\n"
                    f"{top_doc.page_content.strip()[:300]}...\n\n"
                    f"(Note: Set GROQ_API_KEY in .env to activate live Groq LLM inference.)"
                )
            else:
                response = await self.llm.ainvoke(prompt_messages)
                raw_answer = response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            logger.error(f"Error during LLM generation: {e}", exc_info=True)
            raw_answer = (
                "An error occurred while generating the answer from the model. "
                "Please verify your GROQ_API_KEY in .env."
            )

        # 5. Post-generation Output Guardrails
        output_eval = guardrails_service.evaluate_output(raw_answer, context_present=True)
        final_answer = output_eval.sanitized_answer

        # 6. Extract Structured Citations
        citations = extract_citations(top_context_chunks)

        # 7. Assemble Complete Guardrails Report
        report = guardrails_service.create_report(input_eval, output_eval)

        return {
            "answer": final_answer,
            "citations": citations,
            "retrieval": {
                "candidate_count": candidate_count,
                "final_context_count": final_context_count,
            },
            "guardrails": report.model_dump(),
        }


# Singleton service instance
rag_service = RAGService()
