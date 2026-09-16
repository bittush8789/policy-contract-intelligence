"""Pydantic schemas for AI Guardrails evaluation and telemetry."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GuardrailCheckResult(BaseModel):
    """Result of an individual guardrail validator check."""

    check_name: str = Field(..., description="Name of the specific guardrail check")
    passed: bool = Field(..., description="Whether the check passed without violations")
    details: Optional[str] = Field(default=None, description="Diagnostic explanation or trigger message")


class InputGuardrailResult(BaseModel):
    """Aggregated evaluation result for user query input."""

    allowed: bool = Field(default=True, description="Whether the query is permitted to execute")
    sanitized_query: str = Field(..., description="Input query after PII masking or sanitization")
    blocked_reason: Optional[str] = Field(default=None, description="Reason for rejection if blocked")
    injection_detected: bool = Field(default=False, description="Whether prompt injection/jailbreak was flagged")
    pii_detected: bool = Field(default=False, description="Whether PII was identified and redacted")
    checks: List[GuardrailCheckResult] = Field(default_factory=list, description="Individual validator results")


class OutputGuardrailResult(BaseModel):
    """Aggregated evaluation result for generated answer output."""

    passed: bool = Field(default=True, description="Whether the generated output is safe and compliant")
    sanitized_answer: str = Field(..., description="Answer text after redactions or guardrail sanitization")
    system_prompt_leaked: bool = Field(default=False, description="Whether internal instructions were exposed")
    hallucination_flagged: bool = Field(default=False, description="Whether factual inconsistencies were flagged")
    checks: List[GuardrailCheckResult] = Field(default_factory=list, description="Individual validator results")


class GuardrailReport(BaseModel):
    """Full guardrail execution report delivered with API responses."""

    passed: bool = Field(default=True, description="Overall compliance status")
    input_checks: Dict[str, Any] = Field(default_factory=dict, description="Input guardrail metrics")
    output_checks: Dict[str, Any] = Field(default_factory=dict, description="Output guardrail metrics")
