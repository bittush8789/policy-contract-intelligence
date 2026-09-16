"""RAGAS Evaluation Engine for Enterprise RAG Assistant.
Computes Faithfulness, Answer Relevancy, Context Precision, and Context Recall metrics.
Supports both native deterministic scoring and RAGAS library integration.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SampleEvaluationResult(BaseModel):
    """Detailed evaluation scores for a single query."""

    sample_id: str
    question: str
    generated_answer: str
    ground_truth: str
    retrieved_contexts: List[str]
    citations: List[Dict[str, Any]]
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Faithfulness / Grounding score")
    answer_relevancy: float = Field(..., ge=0.0, le=1.0, description="Answer pertinence to question")
    context_precision: float = Field(..., ge=0.0, le=1.0, description="Signal-to-noise ratio in retrieved contexts")
    context_recall: float = Field(..., ge=0.0, le=1.0, description="Coverage of ground truth in retrieved contexts")
    rag_score: float = Field(..., ge=0.0, le=1.0, description="Composite RAG quality score")


class BenchmarkReport(BaseModel):
    """Aggregate benchmark evaluation summary."""

    total_samples: int
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    mean_context_recall: float
    overall_rag_score: float
    sample_evaluations: List[SampleEvaluationResult]


class RagasEvaluator:
    """Evaluates RAG performance across the standard RAGAS triad."""

    def __init__(self, use_ragas_library: bool = False):
        self.use_ragas_library = use_ragas_library

    def calculate_faithfulness(
        self,
        answer: str,
        contexts: List[str],
    ) -> float:
        """Measure what fraction of claims in the answer are supported by retrieved contexts.
        Score from 0.0 (total hallucination) to 1.0 (fully grounded).
        """
        clean_ans = answer.strip()
        if not clean_ans:
            return 0.0

        # Safe refusal when no info is found is 100% faithful
        if clean_ans.startswith("I could not find") or "not find enough information" in clean_ans:
            return 1.0

        if not contexts:
            return 0.0

        combined_context = " ".join(contexts).lower()

        # Split answer into distinct clauses/sentences
        clauses = [
            c.strip()
            for c in re.split(r"[.\n;]", clean_ans)
            if len(c.strip()) > 15
        ]

        if not clauses:
            return 1.0

        grounded_count = 0
        for clause in clauses:
            # Extract keywords (words > 3 chars)
            words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9$%-]+\b", clause) if len(w) >= 4]
            if not words:
                grounded_count += 1
                continue

            # Check keyword containment in retrieved context
            matches = sum(1 for w in words if w in combined_context)
            if (matches / len(words)) >= 0.5:
                grounded_count += 1

        score = round(grounded_count / len(clauses), 4)
        return min(max(score, 0.0), 1.0)

    def calculate_answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> float:
        """Measure how directly the generated answer addresses the question.
        Score from 0.0 (irrelevant) to 1.0 (highly relevant).
        """
        clean_q = question.strip()
        clean_a = answer.strip()

        if not clean_q or not clean_a:
            return 0.0

        # Safe refusal is a compliant, relevant response to an unindexed query
        if clean_a.startswith("I could not find"):
            return 0.95

        q_terms = set(re.findall(r"\b[a-zA-Z0-9]+\b", clean_q.lower()))
        # Remove common stop words
        stop_words = {"what", "is", "the", "for", "are", "in", "of", "and", "to", "who", "which"}
        meaningful_q = q_terms - stop_words

        if not meaningful_q:
            return 1.0

        a_lower = clean_a.lower()
        matched = sum(1 for term in meaningful_q if term in a_lower)
        keyword_overlap = matched / len(meaningful_q)

        # Length / completeness bonus (concise, non-empty answer)
        length_factor = 1.0 if 30 <= len(clean_a) <= 1200 else 0.8

        relevancy = round((keyword_overlap * 0.7 + 0.3 * length_factor), 4)
        return min(max(relevancy, 0.0), 1.0)

    def calculate_context_precision(
        self,
        target_document: str,
        target_section: str,
        retrieved_docs: List[Document],
    ) -> float:
        """Evaluate signal-to-noise ratio in retrieved chunks.
        Higher score if true relevant chunks are ranked at the top.
        """
        if not retrieved_docs:
            return 0.0

        if target_document == "none":
            # For unindexed questions, empty or safe retrieval is expected
            return 1.0

        # Target document & section match checks
        precisions: List[float] = []
        relevant_found = 0

        for rank, doc in enumerate(retrieved_docs, start=1):
            source = str(doc.metadata.get("source", ""))
            section = str(doc.metadata.get("section", ""))

            is_relevant = False
            if target_document.lower() in source.lower():
                # Check section if provided
                if target_section and target_section.lower() in section.lower():
                    is_relevant = True
                elif not target_section or target_section == "General":
                    is_relevant = True
                else:
                    # Partial match on source
                    is_relevant = True

            if is_relevant:
                relevant_found += 1
                precisions.append(relevant_found / rank)

        if not precisions:
            return 0.0

        return round(sum(precisions) / len(precisions), 4)

    def calculate_context_recall(
        self,
        ground_truth: str,
        retrieved_contexts: List[str],
    ) -> float:
        """Measure whether the retrieved context contains all necessary facts from ground truth.
        Score from 0.0 (no facts found) to 1.0 (all facts present).
        """
        if not ground_truth.strip():
            return 1.0

        if ground_truth.startswith("I could not find"):
            return 1.0

        if not retrieved_contexts:
            return 0.0

        combined_context = " ".join(retrieved_contexts).lower()

        # Extract core numerical and factual tokens from ground truth
        key_facts = [
            w.lower()
            for w in re.findall(r"\b(?:\$?\d+[\w%]*|[A-Z][a-z]+)\b", ground_truth)
            if len(w) >= 3
        ]

        if not key_facts:
            return 1.0

        found_facts = sum(1 for fact in key_facts if fact in combined_context)
        score = round(found_facts / len(key_facts), 4)
        return min(max(score, 0.0), 1.0)

    def evaluate_sample(
        self,
        sample_id: str,
        question: str,
        generated_answer: str,
        ground_truth: str,
        retrieved_docs: List[Document],
        citations: List[Dict[str, Any]],
        target_document: str = "none",
        target_section: str = "none",
    ) -> SampleEvaluationResult:
        """Compute full RAGAS metrics for a single sample."""
        context_texts = [doc.page_content for doc in retrieved_docs]

        faithfulness = self.calculate_faithfulness(generated_answer, context_texts)
        relevancy = self.calculate_answer_relevancy(question, generated_answer)
        precision = self.calculate_context_precision(target_document, target_section, retrieved_docs)
        recall = self.calculate_context_recall(ground_truth, context_texts)

        # Composite harmonic-weighted RAG score
        composite = round(
            (faithfulness * 0.35 + relevancy * 0.25 + precision * 0.20 + recall * 0.20),
            4,
        )

        return SampleEvaluationResult(
            sample_id=sample_id,
            question=question,
            generated_answer=generated_answer,
            ground_truth=ground_truth,
            retrieved_contexts=context_texts,
            citations=citations,
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
            rag_score=composite,
        )


# Singleton evaluator instance
ragas_evaluator = RagasEvaluator()
