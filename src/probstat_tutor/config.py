"""Application configuration loaded from environment variables."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    """Validated settings shared by the UI and future business modules."""

    app_title: str = "概率统计 × Python 数据分析学习诊断智能体"
    openai_api_key: SecretStr | None = Field(
        default_factory=lambda: _optional_secret("OPENAI_API_KEY")
    )
    openai_model: str | None = Field(default_factory=lambda: _optional_text("OPENAI_MODEL"))
    questions_path: Path = PROJECT_ROOT / "data" / "questions.yaml"
    curriculum_catalog_path: Path = PROJECT_ROOT / "data" / "curriculum_catalog.yaml"
    rag_manifest_path: Path = PROJECT_ROOT / "data" / "rag" / "manifest.yaml"
    session_db_path: Path = PROJECT_ROOT / "data" / "sessions.sqlite3"
    learning_state_db_path: Path = PROJECT_ROOT / "data" / "learning_state.sqlite3"
    fault_log_path: Path = PROJECT_ROOT / "data" / "logs" / "faults.jsonl"
    model_timeout_seconds: float = Field(
        default_factory=lambda: _optional_float("MODEL_TIMEOUT_SECONDS", 8.0),
        gt=0.0,
        le=120.0,
    )
    model_max_attempts: int = Field(
        default_factory=lambda: _optional_int("MODEL_MAX_ATTEMPTS", 2),
        ge=1,
        le=3,
    )
    model_retry_base_delay_seconds: float = Field(
        default_factory=lambda: _optional_float("MODEL_RETRY_BASE_DELAY_SECONDS", 0.05),
        ge=0.0,
        le=2.0,
    )
    model_circuit_failure_threshold: int = Field(
        default_factory=lambda: _optional_int("MODEL_CIRCUIT_FAILURE_THRESHOLD", 3),
        ge=1,
        le=20,
    )
    model_circuit_open_seconds: float = Field(
        default_factory=lambda: _optional_float("MODEL_CIRCUIT_OPEN_SECONDS", 30.0),
        gt=0.0,
        le=600.0,
    )

    @property
    def has_openai_api_key(self) -> bool:
        """Return whether a non-empty API key is configured."""

        return self.openai_api_key is not None


def _optional_text(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _optional_secret(name: str) -> SecretStr | None:
    value = _optional_text(name)
    return SecretStr(value) if value else None


def _optional_float(name: str, default: float) -> float:
    value = _optional_text(name)
    return default if value is None else float(value)


def _optional_int(name: str, default: int) -> int:
    value = _optional_text(name)
    return default if value is None else int(value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build settings once per application process."""

    return Settings()
