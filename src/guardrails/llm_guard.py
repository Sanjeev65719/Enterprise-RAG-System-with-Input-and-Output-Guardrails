import os
from typing import Tuple, Optional
from openai import OpenAI
from src.core.logger import logger

class LLMGuard:
    """
    Utility class for performing high-assurance binary validation checks using an LLM.
    Returns a boolean indicating if the check passed and a reason for the decision.
    """
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY environment variable not set.")
            raise ValueError("OPENAI_API_KEY is required for LLMGuard.")

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized LLMGuard with model: {self.model_name}")

    def validate(self, system_prompt: str, user_input: str, context: Optional[str] = None) -> Tuple[bool, str]:
        """
        Performs a binary validation check.

        Args:
            system_prompt: The instructions for the LLM on how to validate.
            user_input: The text to be validated (e.g., query or response).
            context: Optional context to use for validation (e.g., retrieved documents).

        Returns:
            A tuple of (allowed, reason).
        """
        full_user_prompt = user_input
        if context:
            full_user_prompt = f"CONTEXT:\n{context}\n\nINPUT TO VALIDATE:\n{user_input}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_user_prompt}
                ],
                temperature=0,
            )

            content = response.choices[0].message.content.strip()

            # Expecting the LLM to start with "YES" or "NO"
            if content.upper().startswith("YES"):
                # Extract reason if present
                reason = content[3:].strip(": \n") if len(content) > 3 else "Check passed."
                return True, reason
            elif content.upper().startswith("NO"):
                reason = content[2:].strip(": \n") if len(content) > 2 else "Check failed."
                return False, reason
            else:
                logger.warning(f"LLMGuard received non-standard response: {content}")
                return False, f"Invalid LLM response format: {content}"

        except Exception as e:
            logger.error(f"LLMGuard API call failed: {str(e)}")
            return False, f"Internal Error during validation: {str(e)}"
