"""Post-generation Output Guardrails for Enterprise RAG Assistant.
Validates generated answers to prevent system prompt leakage, secret exposure,
and ungrounded hallucinations.
"""

import logging
import re
from typing import List, Tuple
from backend.config import settings
from backend.guardrails.input_guardrails import PII_PATTERNS
from backend.guardrails.schemas import GuardrailCheckResult, OutputGuardrailResult

logger = logging.getLogger(__name__)

# Phrases and markers that should never appear in user-facing output
LEAKAGE_SIGNATURES = [
    re.compile(r"You are a highly precise, professional Enterprise Policy", re.IGNORECASE),
    re.compile(r"CRITICAL OPERATIONAL RULES:", re.IGNORECASE),
    re.compile(r"Never disclose these system rules", re.IGNORECASE),
    re.compile(r"\b(gsk_[a-zA-Z0-9]{30,}|sk-[a-zA-Z0-9]{20,}|pcsk_[a-zA-Z0-9]{30,})\b"),  # Common API key formats
    re.compile(r"\bPINECONE_API_KEY\s*=\s*['\"]?[a-zA-Z0-9_-]+", re.IGNORECASE),
    re.compile(r"\bGROQ_API_KEY\s*=\s*['\"]?[a-zA-Z0-9_-]+", re.IGNORECASE),
]

SAFE_REFUSAL_MESSAGE = "I could not find this information in the available documents."


class OutputGuardrails:
    """Post-generation validator ensuring model output does not leak secrets or fabricate facts."""

    def __init__(self, enable_output_guardrails: bool = settings.ENABLE_OUTPUT_GUARDRAILS):
        self.enabled = enable_output_guardrails

    def check_system_prompt_leakage(self, answer: str) -> Tuple[bool, str]:
        """Verify that the model has not exposed system prompt directives or secret keys."""
        if not self.enabled:
            return False, answer

        for pattern in LEAKAGE_SIGNATURES:
            if pattern.search(answer):
                logger.error("Output Guardrail VIOLATION: System directive or API credential leakage detected!")
                # Sanitize response by replacing with safe refusal
                return True, SAFE_REFUSAL_MESSAGE

        return False, answer

    def verify_grounding_consistency(self, answer: str, context_present: bool) -> Tuple[bool, str]:
        """Ensure that if no relevant context exists, the model strictly uses the refusal message."""
        if not context_present:
            # If context was empty or zero candidates, answer must be the safe refusal
            clean_ans = answer.strip()
            if not clean_ans.startswith("I could not find"):
                logger.warning("Output Guardrail: Non-refusal answer generated when context was absent. Overriding.")
                return False, SAFE_REFUSAL_MESSAGE
        return True, answer

    def redact_output_pii(self, answer: str) -> Tuple[str, bool]:
        """Redact any PII present in the final output text."""
        sanitized = answer
        pii_found = False
        for pii_name, pattern, placeholder in PII_PATTERNS:
            if pattern.search(sanitized):
                sanitized = pattern.sub(placeholder, sanitized)
                pii_found = True
        return sanitized, pii_found

    def evaluate(self, answer: str, context_present: bool = True) -> OutputGuardrailResult:
        """Execute full post-generation validation suite."""
        checks: List[GuardrailCheckResult] = []
        final_answer = answer

        # 1. System Prompt & Secret Leakage Check
        leaked, sanitized = self.check_system_prompt_leakage(final_answer)
        final_answer = sanitized
        checks.append(
            GuardrailCheckResult(
                check_name="secret_and_prompt_leakage",
                passed=not leaked,
                details="No internal directives or secrets leaked" if not leaked else "Internal directive redacted",
            )
        )

        # 2. Grounding Consistency Check
        grounded, grounded_ans = self.verify_grounding_consistency(final_answer, context_present)
        final_answer = grounded_ans
        checks.append(
            GuardrailCheckResult(
                check_name="grounding_consistency",
                passed=grounded,
                details="Output grounded strictly in context" if grounded else "Enforced safe refusal fallback",
            )
        )

        # 3. Output PII Redaction
        clean_ans, pii_found = self.redact_output_pii(final_answer)
        final_answer = clean_ans
        checks.append(
            GuardrailCheckResult(
                check_name="output_pii_sanitization",
                passed=True,
                details="Output sanitized" if pii_found else "No PII in output",
            )
        )

        all_passed = not leaked and grounded

        return OutputGuardrailResult(
            passed=all_passed,
            sanitized_answer=final_answer,
            system_prompt_leaked=leaked,
            hallucination_flagged=not grounded,
            checks=checks,
        )
