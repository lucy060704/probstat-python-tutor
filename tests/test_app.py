"""Minimal Streamlit smoke test for the single-page MVP."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_page_renders_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=20)

    assert not app.exception
    assert app.title[0].value.startswith("概率统计")
    assert {button.label for button in app.button} >= {
        "请求提示",
        "提交",
        "下一题",
        "重置演示学习者",
    }
