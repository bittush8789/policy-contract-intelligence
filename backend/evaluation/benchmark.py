"""Benchmark execution script for Enterprise RAG Assistant.
Runs evaluation against the golden dataset, calculates RAGAS metrics,
prints a diagnostic scorecard, and saves benchmark_report.json.

Usage:
    python -m backend.evaluation.benchmark
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from backend.chains.rag_chain import rag_service
from backend.evaluation.dataset import load_golden_dataset
from backend.evaluation.evaluator import BenchmarkReport, SampleEvaluationResult, ragas_evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluation.benchmark")

REPORT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "benchmark_report.json"


async def run_benchmark(save_report: bool = True) -> BenchmarkReport:
    """Execute the full RAG benchmark over the golden dataset."""
    samples = load_golden_dataset()
    logger.info("=" * 70)
    logger.info(f"Starting RAGAS Benchmark Evaluation across {len(samples)} golden queries")
    logger.info("=" * 70)

    sample_evaluations: List[SampleEvaluationResult] = []

    for sample in samples:
        logger.info(f"Evaluating [{sample.id}] '{sample.question}'...")

        # 1. Retrieve candidates & top chunks using hybrid retriever & reranker
        candidate_chunks = rag_service.hybrid_retriever.retrieve(query=sample.question)
        top_chunks = rag_service.reranker.rerank(query=sample.question, documents=candidate_chunks)

        # 2. Run full question answering
        response = await rag_service.answer_question(sample.question)
        generated_ans = response.get("answer", "")
        citations = response.get("citations", [])

        # 3. Evaluate using RAGAS metrics
        result = ragas_evaluator.evaluate_sample(
            sample_id=sample.id,
            question=sample.question,
            generated_answer=generated_ans,
            ground_truth=sample.ground_truth,
            retrieved_docs=top_chunks,
            citations=citations,
            target_document=sample.target_document,
            target_section=sample.target_section,
        )
        sample_evaluations.append(result)

        logger.info(
            f"-> [{sample.id}] Faithfulness: {result.faithfulness:.2f} | "
            f"Relevancy: {result.answer_relevancy:.2f} | "
            f"Precision: {result.context_precision:.2f} | "
            f"Recall: {result.context_recall:.2f} | "
            f"Composite: {result.rag_score:.2f}"
        )

    # Calculate aggregate scores
    n = len(sample_evaluations)
    mean_faith = round(sum(s.faithfulness for s in sample_evaluations) / n, 4) if n else 0.0
    mean_relevancy = round(sum(s.answer_relevancy for s in sample_evaluations) / n, 4) if n else 0.0
    mean_precision = round(sum(s.context_precision for s in sample_evaluations) / n, 4) if n else 0.0
    mean_recall = round(sum(s.context_recall for s in sample_evaluations) / n, 4) if n else 0.0
    overall = round(
        (mean_faith * 0.35 + mean_relevancy * 0.25 + mean_precision * 0.20 + mean_recall * 0.20),
        4,
    )

    report = BenchmarkReport(
        total_samples=n,
        mean_faithfulness=mean_faith,
        mean_answer_relevancy=mean_relevancy,
        mean_context_precision=mean_precision,
        mean_context_recall=mean_recall,
        overall_rag_score=overall,
        sample_evaluations=sample_evaluations,
    )

    if save_report:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved benchmark report to {REPORT_PATH}")

    # Display Summary Scorecard
    print("\n" + "=" * 70)
    print("                RAGAS BENCHMARK SCORECARD")
    print("=" * 70)
    print(f"Total Evaluated Queries: {report.total_samples}")
    print(f"[+] Faithfulness (Grounding):   {report.mean_faithfulness * 100:.1f}%")
    print(f"[+] Answer Relevancy:           {report.mean_answer_relevancy * 100:.1f}%")
    print(f"[+] Context Precision:          {report.mean_context_precision * 100:.1f}%")
    print(f"[+] Context Recall:             {report.mean_context_recall * 100:.1f}%")
    print("-" * 70)
    print(f"[*] OVERALL RAGAS SCORE:        {report.overall_rag_score * 100:.1f}%")
    print("=" * 70 + "\n")

    return report


def main():
    asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
