"""Golden evaluation dataset management for Enterprise RAG assessment.
Maintains curated question, ground truth answer, and source document tuples.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation"
DATASET_PATH = DATA_DIR / "golden_dataset.json"


class EvaluationSample(BaseModel):
    """Structured evaluation sample with query and ground truth."""

    id: str = Field(..., description="Unique sample identifier")
    question: str = Field(..., description="Evaluation question")
    ground_truth: str = Field(..., description="Factually verified reference answer")
    target_document: str = Field(..., description="Expected source document filename")
    target_section: str = Field(..., description="Expected document section")
    category: str = Field(..., description="Category: policy, contract, or unindexed")


# Curated Golden Benchmark Dataset
GOLDEN_EVALUATION_DATA: List[Dict[str, Any]] = [
    {
        "id": "eval_01",
        "question": "What is the annual leave entitlement?",
        "ground_truth": "All full-time permanent employees are entitled to 25 days of paid annual leave per calendar year, accruing on a monthly pro-rata basis.",
        "target_document": "leave_policy.pdf",
        "target_section": "Section 1: Annual Leave Entitlement",
        "category": "policy",
    },
    {
        "id": "eval_02",
        "question": "What is the termination notice period?",
        "ground_truth": "Either party may terminate this agreement without cause upon thirty (30) days prior written notice to the other party.",
        "target_document": "vendor_contract.pdf",
        "target_section": "Section 3: Term & Termination Notice Period",
        "category": "contract",
    },
    {
        "id": "eval_03",
        "question": "Who is eligible for remote work?",
        "ground_truth": "Regular full-time employees who have completed at least ninety (90) days of continuous service with satisfactory performance evaluations are eligible to apply.",
        "target_document": "remote_work_policy.pdf",
        "target_section": "Section 1: Eligibility & Application",
        "category": "policy",
    },
    {
        "id": "eval_04",
        "question": "What are the corporate password requirements?",
        "ground_truth": "All corporate accounts require passwords with a minimum length of 14 characters, combining uppercase letters, lowercase letters, numbers, and special symbols. Multi-Factor Authentication (MFA) is strictly mandatory.",
        "target_document": "information_security_policy.pdf",
        "target_section": "Section 1: Password & Authentication Standards",
        "category": "policy",
    },
    {
        "id": "eval_05",
        "question": "What approval is required for procurement over $50,000?",
        "ground_truth": "Purchases exceeding $50,000 require Vice President and Chief Financial Officer (CFO) approval along with a minimum of three competitive bids submitted to the Procurement Committee.",
        "target_document": "procurement_policy.pdf",
        "target_section": "Section 2: Approval Thresholds & Competitive Bidding",
        "category": "policy",
    },
    {
        "id": "eval_06",
        "question": "What is the customer data retention period?",
        "ground_truth": "Customer and transaction records are retained for seven (7) years following contract termination to comply with statutory and audit obligations.",
        "target_document": "privacy_policy.pdf",
        "target_section": "Section 3: Data Retention & Secure Disposal",
        "category": "policy",
    },
    {
        "id": "eval_07",
        "question": "What is the company stock option vesting schedule?",
        "ground_truth": "I could not find this information in the available documents.",
        "target_document": "none",
        "target_section": "none",
        "category": "unindexed",
    },
]


def ensure_dataset_file(filepath: Path = DATASET_PATH) -> Path:
    """Ensure the golden evaluation dataset JSON exists on disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(GOLDEN_EVALUATION_DATA, f, indent=2, ensure_ascii=False)
        logger.info(f"Initialized golden evaluation dataset at {filepath}")
    return filepath


def load_golden_dataset(filepath: Path = DATASET_PATH) -> List[EvaluationSample]:
    """Load the golden evaluation dataset from disk."""
    ensure_dataset_file(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return [EvaluationSample(**item) for item in raw_data]
