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
    """Verifies that the response is grounded in the provided context (NLI)."""
    def __init__(self):
        self.llm = LLMGuard()

    def check(self, response: str, context: List[str]) -> GuardrailResult:
        logger.info("Running Fact-Checker Agent...")
        context_text = "\n\n".join(context)
        system_prompt = (
            "You are a high-assurance auditor. Your task is to verify if the response is faithfully grounded "
            "in the provided context. Every claim in the response must be explicitly supported by the context. "
            "If the response contains information not found in the context, it is a hallucination. "
            "Answer YES if the response is faithful, and NO if it contains hallucinations. "
            "Provide a concise reason starting with the answer (e.g., 'NO: The claim about X is not in the context')."
        )

        allowed, reason = self.llm.validate(system_prompt, response, context=context_text)

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
        # In a real scenario, the query would be passed here. Since the base class
        # signature doesn't include the query, we rely on the LLM to judge the response's utility.
        system_prompt = (
            "You are a relevance auditor. Your task is to determine if the response "
            "actually answers a likely user query in a helpful and complete manner. "
            "Answer YES if the response is relevant and helpful, and NO if it is generic, "
            "too brief, or fails to provide a substantial answer. "
            "Provide a concise reason starting with the answer (e.g., 'NO: Response is too generic')."
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
