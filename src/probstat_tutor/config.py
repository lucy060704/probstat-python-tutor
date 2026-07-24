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
    session_db_path: Path = PROJECT_ROOT / "data" / "sessions.sqlite3"
    learning_state_db_path: Path = PROJECT_ROOT / "data" / "learning_state.sqlite3"

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build settings once per application process."""

    return Settings()
