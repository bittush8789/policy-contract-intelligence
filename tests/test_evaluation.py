from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from backend.evaluation.dataset import load_golden_dataset, EvaluationSample
from backend.evaluation.evaluator import BenchmarkReport, RagasEvaluator, SampleEvaluationResult
from backend.main import app

client = TestClient(app)


def test_golden_dataset_structure():
    """Verify that the golden benchmark dataset is well-formed across all 7 corporate documents."""
    samples = load_golden_dataset()
    assert len(samples) >= 7, f"Expected at least 7 golden queries, found {len(samples)}"

    for s in samples:
        assert isinstance(s, EvaluationSample)
        assert s.id.startswith("eval_")
        assert len(s.question.strip()) > 5
        assert len(s.ground_truth.strip()) > 5
        assert len(s.target_document.strip()) > 0
        assert len(s.target_section.strip()) > 0


def test_faithfulness_grounded_answer():
    """Verify faithfulness metric scores highly when all claims match retrieved context."""
    evaluator = RagasEvaluator()

    context = [
        "Employees are entitled to 25 days of paid annual leave per calendar year.",
        "Annual leave accrues on a monthly pro-rata basis.",
    ]
    answer = "Employees receive 25 days of paid annual leave per calendar year, accrued monthly on a pro-rata basis."

    score = evaluator.calculate_faithfulness(answer=answer, contexts=context)
    assert score >= 0.8, f"Expected high faithfulness score for grounded answer, got {score}"


def test_faithfulness_hallucinated_answer():
    """Verify faithfulness metric penalizes claims unsupported by the context."""
    evaluator = RagasEvaluator()

    context = [
        "The security policy mandates changing passwords every 90 days with a minimum length of 12 characters."
    ]
    answer = "Employees must attend mandatory underwater basket weaving and purchase a $5,000 gaming laptop."

    score = evaluator.calculate_faithfulness(answer=answer, contexts=context)
    assert score <= 0.4, f"Expected low faithfulness score for hallucinated answer, got {score}"


def test_faithfulness_safe_refusal():
    """Verify that a safe refusal response receives a perfect 1.0 grounding score."""
    evaluator = RagasEvaluator()
    answer = "I could not find enough information in the provided enterprise documents to answer your question."
    score = evaluator.calculate_faithfulness(answer=answer, contexts=[])
    assert score == 1.0


def test_answer_relevancy():
    """Verify answer relevancy metric differentiates relevant vs irrelevant responses."""
    evaluator = RagasEvaluator()

    question = "What is the termination notice period?"
    relevant_answer = "The termination notice period required by the agreement is 30 calendar days."
    irrelevant_answer = "The cafeteria serves pizza on Tuesdays and burgers on Thursdays."

    score_relevant = evaluator.calculate_answer_relevancy(question, relevant_answer)
    score_irrelevant = evaluator.calculate_answer_relevancy(question, irrelevant_answer)

    assert score_relevant > score_irrelevant
    assert score_relevant >= 0.7


def test_context_precision_and_recall():
    """Verify context precision and recall calculations against retrieved documents."""
    evaluator = RagasEvaluator()

    docs = [
        Document(
            page_content="Under Clause 8, the termination notice period is 30 calendar days for either party.",
            metadata={"source": "Master_Services_Agreement.pdf", "section": "Clause 8"},
        ),
        Document(
            page_content="Invoices must be paid within 45 days of receipt.",
            metadata={"source": "Vendor_Contract_SLA.pdf", "section": "Payment Terms"},
        ),
    ]

    # Target document is Master_Services_Agreement.pdf
    precision = evaluator.calculate_context_precision(
        target_document="Master_Services_Agreement.pdf",
        target_section="Clause 8",
        retrieved_docs=docs,
    )
    assert precision > 0.5, f"Expected high precision for top-ranked relevant document, got {precision}"

    # Recall for ground truth facts
    ground_truth = "Either party may terminate by providing 30 calendar days written notice."
    contexts = [d.page_content for d in docs]
    recall = evaluator.calculate_context_recall(ground_truth=ground_truth, retrieved_contexts=contexts)
    assert recall >= 0.7, f"Expected high recall for presence of '30 calendar days', got {recall}"


def test_evaluate_sample_composite():
    """Verify complete sample evaluation computes all RAGAS metrics within [0.0, 1.0]."""
    evaluator = RagasEvaluator()

    docs = [
        Document(
            page_content="Remote work is permitted up to 3 days per week upon manager approval.",
            metadata={"source": "Remote_Work_Policy.pdf", "section": "Section 2"},
        )
    ]

    res = evaluator.evaluate_sample(
        sample_id="test_001",
        question="How many days can employees work remotely?",
        generated_answer="Employees can work remotely up to 3 days per week with manager approval.",
        ground_truth="Employees can work remotely up to 3 days per week with manager approval.",
        retrieved_docs=docs,
        citations=[{"document": "Remote_Work_Policy.pdf", "page": 1, "section": "Section 2"}],
        target_document="Remote_Work_Policy.pdf",
        target_section="Section 2",
    )

    assert isinstance(res, SampleEvaluationResult)
    assert 0.0 <= res.faithfulness <= 1.0
    assert 0.0 <= res.answer_relevancy <= 1.0
    assert 0.0 <= res.context_precision <= 1.0
    assert 0.0 <= res.context_recall <= 1.0
    assert 0.0 <= res.rag_score <= 1.0
    assert res.rag_score >= 0.75


def test_evaluation_benchmark_endpoint():
    """Test GET /api/evaluation/benchmark returns valid BenchmarkReport."""
    mock_report = BenchmarkReport(
        total_samples=1,
        mean_faithfulness=0.95,
        mean_answer_relevancy=0.92,
        mean_context_precision=0.90,
        mean_context_recall=1.0,
        overall_rag_score=0.94,
        sample_evaluations=[
            SampleEvaluationResult(
                sample_id="eval_01",
                question="What is the annual leave entitlement?",
                generated_answer="Employees are entitled to 25 days paid annual leave.",
                ground_truth="Employees are entitled to 25 days paid annual leave.",
                retrieved_contexts=["25 days annual leave."],
                citations=[{"document": "leave_policy.pdf", "page": 1, "section": "Section 1"}],
                faithfulness=0.95,
                answer_relevancy=0.92,
                context_precision=0.90,
                context_recall=1.0,
                rag_score=0.94,
            )
        ],
    )

    with patch("backend.api.evaluation.run_benchmark", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_report
        with patch("backend.api.evaluation.REPORT_PATH", new=Path("non_existent_report_12345.json")):
            response = client.get("/api/evaluation/benchmark")
            assert response.status_code == 200

            data = response.json()
            assert data["total_samples"] == 1
            assert data["mean_faithfulness"] == 0.95
            assert data["mean_answer_relevancy"] == 0.92
            assert data["mean_context_precision"] == 0.90
            assert data["mean_context_recall"] == 1.0
            assert data["overall_rag_score"] == 0.94
            assert len(data["sample_evaluations"]) == 1
            assert data["sample_evaluations"][0]["sample_id"] == "eval_01"


def test_evaluation_run_endpoint():
    """Test POST /api/evaluation/run executes benchmark and returns report."""
    mock_report = BenchmarkReport(
        total_samples=1,
        mean_faithfulness=1.0,
        mean_answer_relevancy=1.0,
        mean_context_precision=1.0,
        mean_context_recall=1.0,
        overall_rag_score=1.0,
        sample_evaluations=[],
    )

    with patch("backend.api.evaluation.run_benchmark", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_report
        response = client.post("/api/evaluation/run")
        assert response.status_code == 200

        data = response.json()
        assert data["total_samples"] == 1
        assert data["overall_rag_score"] == 1.0
        assert mock_run.called

