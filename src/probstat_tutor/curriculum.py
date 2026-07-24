"""Load and validate the versioned YAML question bank."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from probstat_tutor.config import get_settings
from probstat_tutor.schemas import QuestionBank


class CurriculumLoadError(ValueError):
    """A human-readable question bank loading error."""


def load_question_bank(path: str | Path) -> QuestionBank:
    """Load one YAML file and validate all curriculum data with Pydantic."""

    question_path = Path(path)
    try:
        content = question_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise CurriculumLoadError(f"无法加载题库：文件不存在：{question_path}") from error
    except OSError as error:
        raise CurriculumLoadError(f"无法读取题库文件 {question_path}：{error}") from error

    try:
        raw_data = yaml.safe_load(content)
    except yaml.YAMLError as error:
        raise CurriculumLoadError(f"题库 YAML 格式错误（{question_path}）：{error}") from error

    if not isinstance(raw_data, dict):
        raise CurriculumLoadError(f"题库内容必须是一个对象（{question_path}）")

    try:
        return QuestionBank.model_validate(raw_data)
    except ValidationError as error:
        raise CurriculumLoadError(
            f"题库内容未通过 Pydantic 验证（{question_path}）：\n{error}"
        ) from error


def load_default_question_bank() -> QuestionBank:
    """Load the question bank configured for this application."""

    return load_question_bank(get_settings().questions_path)
