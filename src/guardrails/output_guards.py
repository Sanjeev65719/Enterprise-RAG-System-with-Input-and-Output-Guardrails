import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List

from src.core.logger import logger
from src.guardrails.llm_guard import LLMGuard


@dataclass
class GuardrailResult:
    allowed: bool
    reason: Optional[str] = None
    suggestion: Optional[str] = None

class OutputGuardrail:
    """Base class for output guardrails."""
    def check(self, response: str, context: List[str]) -> GuardrailResult:
        raise NotImplementedError

class FactCheckerAgent(OutputGuardrail):
    """Verifies that the response is faithfully grounded or factually/mathematically accurate."""
    def __init__(self):
        self.llm = LLMGuard()

    def check(self, response: str, context: List[str]) -> GuardrailResult:
        logger.info("Running Fact-Checker Agent...")
        active_chunks = [c for c in context if c and c.strip()]
        context_text = "\n\n".join(active_chunks)

        if active_chunks:
            system_prompt = (
                "You are a high-assurance auditor. Your task is to verify the factual grounding of the response:\n"
                "1. If the response makes claims regarding enterprise policies, internal processes, or organization data, "
                "those claims must be faithfully grounded in the provided context.\n"
                "2. If the response provides general arithmetic, mathematical calculations, scientific principles, "
                "or general technical reasoning, verify that it is mathematically and factually accurate and does not "
                "hallucinate fake internal enterprise records.\n"
                "Answer YES if the response is faithful and factually sound. "
                "Answer NO if it contains hallucinations or false claims.\n"
                "Provide a concise reason starting with the answer (e.g., 'NO: Claim X is contradicted or ungrounded')."
            )
            allowed, reason = self.llm.validate(system_prompt, response, context=context_text)
        else:
            system_prompt = (
                "You are a high-assurance auditor. Context is empty because this is a general arithmetic, "
                "scientific, or general factual query. Verify that the response is mathematically and factually accurate, "
                "coherent, and does not fabricate fictitious internal enterprise records.\n"
                "Answer YES if the response is accurate and coherent. "
                "Answer NO if it contains mathematical errors, hallucinations, or falsehoods.\n"
                "Provide a concise reason starting with the answer (e.g., 'NO: Arithmetic calculation is incorrect')."
            )
            allowed, reason = self.llm.validate(system_prompt, response)

        if not allowed:
            return GuardrailResult(allowed=False, reason=reason)

        return GuardrailResult(allowed=True)


class RelevanceAgent(OutputGuardrail):
    """Ensures the response actually answers the user's query."""
    def __init__(self):
        self.llm = LLMGuard()

    def check(self, response: str, context: List[str]) -> GuardrailResult:
        logger.info("Running Relevance Agent...")
        # Note: The context is passed for reference, but the primary check is response vs query.
        system_prompt = (
            "You are a relevance auditor. Your task is to determine if the response "
            "answers a likely user query in an accurate, helpful, and appropriate manner. "
            "Concise, direct answers to arithmetic, scientific, or factual queries (e.g., mathematical results) are acceptable. "
            "Answer YES if the response is relevant and helpful, and NO if it is evasive, completely off-topic, "
            "or meaningless gibberish. "
            "Provide a concise reason starting with the answer (e.g., 'NO: Response is off-topic')."
        )

        allowed, reason = self.llm.validate(system_prompt, response)

        if not allowed:
            return GuardrailResult(allowed=False, reason=reason)

        return GuardrailResult(allowed=True)


class ComplianceAgent(OutputGuardrail):
    """Ensures tone and compliance with enterprise standards."""
    def __init__(self):
        self.llm = LLMGuard()

    def check(self, response: str, context: List[str]) -> GuardrailResult:
        logger.info("Running Compliance Agent...")
        system_prompt = (
            "You are a compliance auditor. Your task is to ensure the response follows "
            "enterprise standards and does not leak internal secrets (e.g., passwords, "
            "internal-only keys, or unauthorized sensitive data). "
            "Answer YES if the response is compliant, and NO if it violates compliance. "
            "Provide a concise reason starting with the answer (e.g., 'NO: Leaked a password')."
        )

        allowed, reason = self.llm.validate(system_prompt, response)

        if not allowed:
            return GuardrailResult(allowed=False, reason=reason)

        return GuardrailResult(allowed=True)


class OutputGuardrailOrchestrator:
    """Coordinates all output guardrails."""
    def __init__(self):
        self.guards = [
            FactCheckerAgent(),
            RelevanceAgent(),
            ComplianceAgent()
        ]

    def verify(self, response: str, context: List[str]) -> Tuple[bool, str]:
        for guard in self.guards:
            result = guard.check(response, context)
            if not result.allowed:
                logger.warning(f"Output Guardrail failed: {result.reason}")
                return False, result.reason

        return True, "Verified"

if __name__ == "__main__":
    orchestrator = OutputGuardrailOrchestrator()

    # Test 1: Good response
    ok, res = orchestrator.verify("The security policy states that PII must be encrypted.", ["PII must be encrypted"])
    print(f"Test 1: {ok}, {res}")

    # Test 2: Hallucination (empty context)
    ok, res = orchestrator.verify("The password is 12345.", [])
    print(f"Test 2: {ok}, {res}")
