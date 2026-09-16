"""Evaluation API routes for viewing and triggering RAGAS benchmarks."""

import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from backend.evaluation.benchmark import REPORT_PATH, run_benchmark
from backend.evaluation.evaluator import BenchmarkReport

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation & RAGAS"])


@router.get(
    "/benchmark",
    response_model=BenchmarkReport,
    summary="Get Latest RAGAS Benchmark Report",
    description="Retrieves the most recent evaluation scorecard including Faithfulness, Relevancy, Precision, and Recall.",
)
async def get_benchmark_report() -> BenchmarkReport:
    """Return the cached benchmark evaluation report, or execute on first request."""
    if REPORT_PATH.exists():
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BenchmarkReport(**data)
        except Exception as e:
            logger.error(f"Failed to read cached benchmark report: {e}")

    # If no report is cached, generate one
    logger.info("No cached benchmark report found. Executing initial benchmark...")
    report = await run_benchmark(save_report=True)
    return report


@router.post(
    "/run",
    response_model=BenchmarkReport,
    summary="Trigger Fresh RAGAS Benchmark Run",
    description="Executes a new RAGAS evaluation run across the golden enterprise dataset and updates cached metrics.",
)
async def trigger_benchmark_run() -> BenchmarkReport:
    """Execute a fresh benchmark run."""
    try:
        logger.info("API trigger: starting fresh RAGAS benchmark run...")
        report = await run_benchmark(save_report=True)
        return report
    except Exception as e:
        logger.error(f"Error during benchmark execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark execution failed: {str(e)}",
        )
