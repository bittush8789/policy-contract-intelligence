"""Unified AI Guardrails Service orchestrating pre-retrieval and post-generation checks."""

import logging
from typing import Any, Dict, Optional, Tuple
from backend.config import settings
from backend.guardrails.input_guardrails import InputGuardrails
from backend.guardrails.output_guardrails import OutputGuardrails
from backend.guardrails.schemas import GuardrailReport, InputGuardrailResult, OutputGuardrailResult

logger = logging.getLogger(__name__)


class GuardrailsService:
    """Enterprise AI Guardrails Service."""

    def __init__(
        self,
        input_guardrails: Optional[InputGuardrails] = None,
        output_guardrails: Optional[OutputGuardrails] = None,
        enabled: bool = settings.ENABLE_GUARDRAILS,
    ):
        self.enabled = enabled
        self.input_guardrails = input_guardrails or InputGuardrails()
        self.output_guardrails = output_guardrails or OutputGuardrails()

    def evaluate_input(self, query: str) -> InputGuardrailResult:
        """Run pre-retrieval input evaluation."""
        if not self.enabled or not settings.ENABLE_INPUT_GUARDRAILS:
            return InputGuardrailResult(
                allowed=True,
                sanitized_query=query,
                blocked_reason=None,
                injection_detected=False,
                pii_detected=False,
                checks=[],
            )
        return self.input_guardrails.evaluate(query)

    def evaluate_output(self, answer: str, context_present: bool = True) -> OutputGuardrailResult:
        """Run post-generation output evaluation."""
        if not self.enabled or not settings.ENABLE_OUTPUT_GUARDRAILS:
            return OutputGuardrailResult(
                passed=True,
                sanitized_answer=answer,
                system_prompt_leaked=False,
                hallucination_flagged=False,
                checks=[],
            )
        return self.output_guardrails.evaluate(answer, context_present=context_present)

    def create_report(
        self,
        input_result: InputGuardrailResult,
        output_result: Optional[OutputGuardrailResult] = None,
    ) -> GuardrailReport:
        """Assemble a standardized audit report for API consumers."""
        input_metrics = {
            "allowed": input_result.allowed,
            "injection_detected": input_result.injection_detected,
            "pii_redacted": input_result.pii_detected,
            "blocked_reason": input_result.blocked_reason,
        }

        output_metrics = {}
        if output_result is not None:
            output_metrics = {
                "passed": output_result.passed,
                "system_prompt_leaked": output_result.system_prompt_leaked,
                "hallucination_flagged": output_result.hallucination_flagged,
            }

        overall_passed = input_result.allowed and (output_result.passed if output_result else True)

        return GuardrailReport(
            passed=overall_passed,
            input_checks=input_metrics,
            output_checks=output_metrics,
        )


# Singleton guardrails instance
guardrails_service = GuardrailsService()
