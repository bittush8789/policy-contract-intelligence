"""Pre-retrieval Input Guardrails for Enterprise RAG Assistant.
Scans for prompt injections, jailbreak attempts, PII, and malicious inputs before retrieval.
"""

import logging
import re
from typing import List, Tuple
from backend.config import settings
from backend.guardrails.schemas import GuardrailCheckResult, InputGuardrailResult

logger = logging.getLogger(__name__)

# Patterns targeting prompt injection and jailbreak overrides
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules|commands)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules|guidelines)", re.IGNORECASE),
    re.compile(r"(reveal|display|output|show|print|leak)\s+(your\s+)?(system\s+prompt|internal\s+instructions|system\s+directives)", re.IGNORECASE),
    re.compile(r"what\s+(is|are)\s+your\s+(initial\s+prompt|system\s+instructions|secret\s+instructions)", re.IGNORECASE),
    re.compile(r"\b(DAN\s+mode|jailbreak|jailbroken|unrestricted\s+mode)\b", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+developer\s+mode|free\s+of\s+all\s+rules|an\s+unfiltered\s+ai)", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?(content\s+filters|safety\s+filters|security\s+rules)", re.IGNORECASE),
    re.compile(r"\[/?(system|instruction|admin|override)\]", re.IGNORECASE),
    re.compile(r"(---\s*BEGIN\s+SYSTEM|===\s*SYSTEM\s*OVERRIDE)", re.IGNORECASE),
]

# PII Detection and Redaction regexes
PII_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    ("CreditCard", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"), "[REDACTED_CREDIT_CARD]"),
    ("Email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"), "[REDACTED_EMAIL]"),
    ("Phone", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
]

# Harmful & Dangerous Input Patterns
HARMFUL_PATTERNS = [
    re.compile(r"\b(how\s+to\s+(build|make|synthesize)\s+(a\s+bomb|explosives|biological\s+weapon))\b", re.IGNORECASE),
    re.compile(r"\b(write\s+(ransomware|malware|keylogger|exploit\s+code))\b", re.IGNORECASE),
    re.compile(r"\b(ddos\s+attack|sql\s+injection\s+payload\s+for\s+hacking)\b", re.IGNORECASE),
]


class InputGuardrails:
    """Pre-retrieval validator ensuring user queries are safe, policy-compliant, and sanitized."""

    def __init__(
        self,
        enable_injection_detection: bool = settings.BLOCK_PROMPT_INJECTION,
        enable_pii_redaction: bool = settings.ENABLE_PII_REDACTION,
    ):
        self.enable_injection_detection = enable_injection_detection
        self.enable_pii_redaction = enable_pii_redaction

    def check_prompt_injection(self, query: str) -> GuardrailCheckResult:
        """Scan query for prompt injection or jailbreak patterns."""
        if not self.enable_injection_detection:
            return GuardrailCheckResult(check_name="prompt_injection", passed=True)

        for pattern in PROMPT_INJECTION_PATTERNS:
            match = pattern.search(query)
            if match:
                matched_text = match.group(0)
                logger.warning(f"Input Guardrail TRIGGERED: Prompt injection pattern detected: '{matched_text}'")
                return GuardrailCheckResult(
                    check_name="prompt_injection",
                    passed=False,
                    details=f"Adversarial prompt injection pattern detected: '{matched_text}'",
                )

        return GuardrailCheckResult(check_name="prompt_injection", passed=True)

    def check_harmful_intent(self, query: str) -> GuardrailCheckResult:
        """Scan query for prohibited hazardous content."""
        for pattern in HARMFUL_PATTERNS:
            if pattern.search(query):
                logger.warning("Input Guardrail TRIGGERED: Harmful intent pattern detected.")
                return GuardrailCheckResult(
                    check_name="harmful_intent",
                    passed=False,
                    details="Prohibited malicious or harmful query pattern detected.",
                )

        return GuardrailCheckResult(check_name="harmful_intent", passed=True)

    def redact_pii(self, query: str) -> Tuple[str, bool, List[str]]:
        """Identify and redact Personally Identifiable Information (PII)."""
        if not self.enable_pii_redaction:
            return query, False, []

        sanitized = query
        detected_types = []

        for pii_name, pattern, placeholder in PII_PATTERNS:
            if pattern.search(sanitized):
                sanitized = pattern.sub(placeholder, sanitized)
                detected_types.append(pii_name)
                logger.info(f"Input Guardrail: Redacted {pii_name} in user query.")

        pii_found = len(detected_types) > 0
        return sanitized, pii_found, detected_types

    def evaluate(self, query: str) -> InputGuardrailResult:
        """Run all input guardrail checks on the query."""
        checks: List[GuardrailCheckResult] = []

        # 1. Prompt Injection Check
        injection_result = self.check_prompt_injection(query)
        checks.append(injection_result)
        if not injection_result.passed:
            return InputGuardrailResult(
                allowed=False,
                sanitized_query=query,
                blocked_reason="This request was blocked by enterprise AI input guardrails due to detected prompt injection or policy override attempt.",
                injection_detected=True,
                pii_detected=False,
                checks=checks,
            )

        # 2. Harmful Intent Check
        harm_result = self.check_harmful_intent(query)
        checks.append(harm_result)
        if not harm_result.passed:
            return InputGuardrailResult(
                allowed=False,
                sanitized_query=query,
                blocked_reason="This request was blocked by enterprise AI guardrails due to prohibited hazardous content.",
                injection_detected=False,
                pii_detected=False,
                checks=checks,
            )

        # 3. PII Redaction
        sanitized_query, pii_found, pii_types = self.redact_pii(query)
        checks.append(
            GuardrailCheckResult(
                check_name="pii_redaction",
                passed=True,
                details=f"Redacted: {', '.join(pii_types)}" if pii_found else "Clean",
            )
        )

        return InputGuardrailResult(
            allowed=True,
            sanitized_query=sanitized_query,
            blocked_reason=None,
            injection_detected=False,
            pii_detected=pii_found,
            checks=checks,
        )
