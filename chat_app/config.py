import os
from dataclasses import dataclass


def _getenv_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


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

    # Модели по ролям (если не заданы — используем LLM_MODEL).
    condition_model: str = os.getenv("CONDITION_MODEL", "") or llm_model
    judge_model: str = os.getenv("JUDGE_MODEL", "") or llm_model
    revise_model: str = os.getenv("REVISE_MODEL", "") or judge_model
    summary_model: str = os.getenv("SUMMARY_MODEL", "") or llm_model

    # Версия пайплайна чат-агента (0.1 — текущая линейная реализация).
    agent_pipeline_version: str = os.getenv("AGENT_PIPELINE_VERSION", "0.1")

    # SGR converter (plain text -> ScenarioDefinition)
    sgr_model: str = os.getenv("SGR_MODEL", "") or llm_model
    sgr_log_prompts: bool = _getenv_bool("SGR_LOG_PROMPTS", False)


settings = Settings()
