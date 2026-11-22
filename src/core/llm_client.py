import os
from typing import Tuple, Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Lightweight fallback if python-dotenv is not installed
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    if os.path.exists(_env_path):
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v
from openai import OpenAI
from src.core.logger import logger

def get_llm_client_and_model(preferred_model: Optional[str] = None) -> Tuple[OpenAI, str, str]:
    """
    Returns a configured OpenAI-compatible client, the model name, and the provider name.
    Supports Groq, OpenAI, and Ollama.
    
    Priority:
    1. Explicit LLM_PROVIDER ('groq', 'openai', 'ollama')
    2. If GROQ_API_KEY is present and OPENAI_API_KEY is missing/empty, defaults to 'groq'
    3. Defaults to 'openai'
    """
    provider_env = os.getenv("LLM_PROVIDER", "").strip().lower()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if provider_env == "groq" or (groq_key and not openai_key) or (groq_key and provider_env != "openai"):
        provider = "groq"
        api_key = groq_key
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        model = preferred_model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    elif provider_env == "ollama":
        provider = "ollama"
        api_key = "ollama"
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model = preferred_model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    else:
        provider = "openai"
        api_key = openai_key
        base_url = None
        model = preferred_model or os.getenv("OPENAI_MODEL", "gpt-4o")

    if not api_key:
        error_msg = (
            f"No API key provided for LLM provider '{provider}'. "
            "Please set GROQ_API_KEY (for free Groq inference) or OPENAI_API_KEY in your .env file."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model, provider
