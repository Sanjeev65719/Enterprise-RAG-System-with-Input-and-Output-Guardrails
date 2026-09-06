import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from src.core.logger import logger
from src.orchestrator.graph import RAGOrchestrator

@dataclass
class EvalTriplet:
    query: str
    ground_truth: str
    expected_context: List[str]

class RAGEvaluator:
    """Evaluates RAG performance using an LLM-as-a-Judge approach (similar to RAGAS)."""

    def __init__(self, orchestrator: RAGOrchestrator):
        self.orchestrator = orchestrator

    def evaluate_triplet(self, triplet: EvalTriplet) -> Dict[str, float]:
        """Evaluates a single query-response pair across 3 metrics."""
        result = self.orchestrator.run(triplet.query)

        if result['status'] != 'success':
            return {"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0}

        response = result['response']
        context = result['context']

        # Mocked scoring (In production, these would be LLM calls)
        faithfulness = self._score_faithfulness(response, context)
        relevance = self._score_relevance(response, triplet.ground_truth)
        precision = self._score_precision(context, triplet.expected_context)

        return {
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "context_precision": precision
        }

    def _score_faithfulness(self, response: str, context: str) -> float:
        # Mock: check if response keywords are in context
        return 0.9 if context and len(response) > 10 else 0.1

    def _score_relevance(self, response: str, ground_truth: str) -> float:
        # Mock: semantic similarity
        return 0.85 if len(response) > 10 else 0.2

    def _score_precision(self, context: str, expected: List[str]) -> float:
        # Mock: check if expected chunks are in the retrieved context
        return 0.95 if context else 0.0

    def run_benchmark(self, triplets: List[EvalTriplet]) -> Dict[str, float]:
        """Runs benchmark over a dataset and returns average scores."""
        logger.info(f"Running benchmark on {len(triplets)} triplets...")
        total_scores = {"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0}

        for t in triplets:
            scores = self.evaluate_triplet(t)
            for k, v in scores.items():
                total_scores[k] += v

        return {k: v / len(triplets) for k, v in total_scores.items()}

if __name__ == "__main__":
    orchestrator = RAGOrchestrator()
    evaluator = RAGEvaluator(orchestrator)

    dataset = [
        EvalTriplet(
            query="How to mask PII?",
            ground_truth="Use the PII Masking Agent to redact sensitive data.",
            expected_context=["PII Masking Agent", "Redact"]
        ),
        EvalTriplet(
            query="What is the security policy?",
            ground_truth="All data must be encrypted at rest.",
            expected_context=["Security Policy", "Encryption"]
        )
    ]

    final_scores = evaluator.run_benchmark(dataset)
    print(f"Benchmark Results: {final_scores}")
