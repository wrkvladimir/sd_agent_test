import os
from dataclasses import dataclass


@dataclass
class Settings:
    """
    Simple settings holder for the chat service.

    Values are read from environment variables with sensible defaults.
    """

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    retrieval_url: str = os.getenv("RETRIEVAL_URL", "http://localhost:8001")
    scenario_storage_path: str = os.getenv("SCENARIO_STORAGE_PATH", "data")

    # Настройки LLM-провайдера с OpenAI-совместимым API (например, OpenRouter).
    # Переменные называются OPENAI_* только ради совместимости с SDK.
    llm_api_key: str | None = os.getenv("OPENAI_API_KEY")
    llm_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")


settings = Settings()
