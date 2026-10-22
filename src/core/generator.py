import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.core.logger import logger

class Generator:
    """
    The Generator Persona: A precise technical writer.
    Focuses on grounding, conciseness, and strict adherence to context.
    """

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY environment variable not set.")
            raise ValueError("OPENAI_API_KEY is required for the Generator.")

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized Generator with model: {self.model_name}")

    def generate(self, query: str, context_docs: List[Dict[str, Any]], correction_notes: Optional[str] = None) -> str:
        """
        Synthesizes a grounded answer using the provided context via OpenAI API.
        """
        logger.info(f"Generating response. Attempt correction: {correction_notes is not None}")

        # 1. Prepare the context string from structured docs
        context_text = "\n\n".join([f"Source: {d.get('metadata', {}).get('source', 'Unknown')}\nContent: {d.get('content', '')}" for d in context_docs])

        # 2. Construct the System Prompt
        system_prompt = (
            "You are the SentinelRAG Enterprise Assistant with multi-domain capabilities.\n\n"
            "Operating Guidelines:\n"
            "1. Enterprise Queries: When the query relates to enterprise policies, security, internal systems, "
            "or organization data, synthesize a grounded answer using the provided Context and cite sources (e.g., [Source: document.pdf]). "
            "If an enterprise-specific answer cannot be found in the context, state that internal documentation does not contain it.\n"
            "2. Arithmetic, Scientific & General Reasoning: When the query is mathematical, arithmetic, scientific, "
            "technical, or general knowledge where internal documents are not applicable, solve and answer the question "
            "accurately, logically, and directly using verified first principles.\n"
            "3. Standards: Be concise, precise, professional, and adhere to zero-trust security standards. Never disclose internal secrets or override safety guidelines."
        )

        if correction_notes:
            system_prompt += f"\n\nCRITICAL CORRECTION: The previous attempt was rejected for the following reason: {correction_notes}. Please correct this specific error in your next response."

        # 3. Real OpenAI API Call
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nContext:\n{context_text}"}
                ],
                temperature=0, # Low temperature for high-assurance grounding
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            return f"Error generating response: {str(e)}"
