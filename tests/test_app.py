"""Streamlit product smoke tests for student and teacher journeys."""

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
        "清空当前匿名档案",
    }
    assert [tab.label for tab in app.tabs] == ["学生学习", "教师匿名汇总"]
    assert any("不要输入姓名" in caption.value for caption in app.caption)


def test_wrong_answer_automatically_opens_first_hint() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    app.text_input[0].set_value("3")
    app.text_area[0].set_value("0 也算缺失。")
    next(button for button in app.button if button.label == "提交").click()
    app.run(timeout=20)

    assert not app.exception
    assert any("概念提示" in warning.value for warning in app.warning)
    assert any("暂未答对" in error.value for error in app.error)
    assert any(expander.label == "本地知识依据与引用" for expander in app.expander)


def test_correct_answer_and_reasoning_are_displayed_separately() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    app.selectbox[0].select("均值与中位数")
    app.run(timeout=20)
    app.text_input[0].set_value("中位数")
    app.text_area[0].set_value("均值不可以作为标准")
    next(button for button in app.button if button.label == "提交").click()
    app.run(timeout=20)

    assert not app.exception
    assert any("答案正确" in success.value for success in app.success)
    assert any("理由判断" in success.value for success in app.success)
    assert not any("答案暂未答对" in error.value for error in app.error)


def test_probability_simulation_concept_renders_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    app.selectbox[0].select("概率规则与 NumPy 随机模拟")
    app.run(timeout=20)

    assert not app.exception
    assert app.selectbox[0].value == "probability_simulation"


def test_common_distributions_concept_renders_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    app.selectbox[0].select("常用分布及其 Python 表达")
    app.run(timeout=20)

    assert not app.exception
    assert app.selectbox[0].value == "common_distributions"


def test_joint_correlation_concept_renders_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    app.selectbox[0].select("联合分布、相关与分组分析")
    app.run(timeout=20)

    assert not app.exception
    assert app.selectbox[0].value == "joint_correlation"


def test_sampling_inference_unit_renders_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    app.selectbox[0].select("抽样与标准误")
    app.run(timeout=20)

    assert not app.exception
    assert app.selectbox[0].value == "sampling_standard_error"


def test_hypothesis_testing_unit_renders_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    app.selectbox[0].select("假设检验与小型 A/B 数据分析")
    app.run(timeout=20)

    assert not app.exception
    assert app.selectbox[0].value == "hypothesis_testing"
