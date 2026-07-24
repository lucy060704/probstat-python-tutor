"""Single-page Streamlit MVP; all learning logic lives in the service layer."""

import asyncio
import uuid

import streamlit as st

from probstat_tutor.config import get_settings
from probstat_tutor.schemas import ConceptId, DiagnosticReport
from probstat_tutor.service import LearningService, LearningServiceError

CONCEPT_LABELS = {
    ConceptId.MEAN_MEDIAN: "均值与中位数",
    ConceptId.VARIANCE_STD: "方差与标准差",
    ConceptId.SAMPLING_STANDARD_ERROR: "抽样与标准误",
    ConceptId.CONFIDENCE_INTERVAL: "置信区间",
}


@st.cache_resource
def get_learning_service() -> LearningService:
    """Reuse database connections and curriculum configuration across reruns."""

    return LearningService()


settings = get_settings()
service = get_learning_service()

st.set_page_config(page_title=settings.app_title, page_icon="📊", layout="wide")
st.title(settings.app_title)
st.caption("v0.1 单页学习 MVP · 简体中文 · 确定性判题")

if service.offline_mode:
    st.info("当前为离线演示模式：不会调用 OpenAI API，题库、判题、掌握度和推荐仍可使用。")

st.session_state.setdefault("session_id", f"streamlit-{uuid.uuid4().hex}")
st.session_state.setdefault("hint_level", 0)

pending_question_id = st.session_state.pop("pending_question_id", None)
if pending_question_id:
    pending_question = service.get_question(pending_question_id)
    st.session_state["selected_concept"] = pending_question.concept_id
    st.session_state["current_question_id"] = pending_question.id
    st.session_state["selection_signature"] = (
        st.session_state.get("learner_id", "demo"),
        pending_question.concept_id,
    )
    st.session_state["hint_level"] = 0
    st.session_state["report"] = None
    for input_key in ("answer", "reasoning", "python_code"):
        st.session_state[input_key] = ""

left, middle, right = st.columns([1.05, 1.9, 1.25], gap="large")

with left:
    st.subheader("学习状态")
    learner_id = st.text_input("学习者 ID", value="demo", key="learner_id").strip() or "demo"
    concept_id = st.selectbox(
        "知识点",
        options=list(ConceptId),
        format_func=lambda concept: CONCEPT_LABELS[concept],
        key="selected_concept",
    )

    selection_signature = (learner_id, concept_id)
    if st.session_state.get("selection_signature") != selection_signature:
        try:
            selected_question = service.choose_question(learner_id, concept_id)
            st.session_state["current_question_id"] = selected_question.id
        except LearningServiceError as error:
            st.session_state["current_question_id"] = None
            st.info(str(error))
        st.session_state["selection_signature"] = selection_signature
        st.session_state["hint_level"] = 0
        st.session_state["report"] = None

    dashboard = service.get_dashboard(learner_id, concept_id)
    st.markdown("**当前四维掌握度**")
    for dimension, score in dashboard.selected_mastery.items():
        st.caption(f"{dimension}: {score:.2f}")
        st.progress(score)

    st.markdown("**最近学习记录**")
    if dashboard.recent_records:
        st.dataframe(list(dashboard.recent_records), hide_index=True, use_container_width=True)
    else:
        st.caption("还没有学习记录。")

    if st.button("重置演示学习者", type="secondary", use_container_width=True):
        service.reset_demo_learner(learner_id)
        for key in (
            "selection_signature",
            "current_question_id",
            "report",
            "answer",
            "reasoning",
            "python_code",
        ):
            st.session_state.pop(key, None)
        st.session_state["hint_level"] = 0
        st.rerun()

current_question_id = st.session_state.get("current_question_id")
question = service.get_question(current_question_id) if current_question_id else None

with middle:
    st.subheader("当前题目")
    if question is None:
        st.info("当前知识点暂时没有可作答题目，请查看左侧提示或选择其他知识点。")
    else:
        st.markdown(f"### {question.title}")
        st.write(question.prompt)
        st.markdown("**数据预览**")
        st.json(question.dataset, expanded=True)

        answer = st.text_input("答案输入", key="answer", placeholder="请输入答案")
        reasoning = st.text_area(
            "思考过程输入",
            key="reasoning",
            placeholder="请写出你的判断依据或计算步骤",
            height=110,
        )
        python_code = st.text_area(
            "Python 代码输入（仅作为文本分析，不会执行）",
            key="python_code",
            placeholder="可选：粘贴你的 Python 代码",
            height=120,
        )

        hint_column, submit_column = st.columns(2)
        with hint_column:
            if st.button("请求提示", use_container_width=True):
                st.session_state["hint_level"] = min(
                    3, st.session_state["hint_level"] + 1
                )
        with submit_column:
            submitted = st.button("提交", type="primary", use_container_width=True)

        if st.session_state["hint_level"] > 0:
            st.warning(service.get_hint(question.id, st.session_state["hint_level"]))

        if submitted:
            if not answer.strip():
                st.warning("请先填写答案，再提交。")
            else:
                try:
                    report = asyncio.run(
                        service.submit(
                            learner_id=learner_id,
                            session_id=st.session_state["session_id"],
                            question_id=question.id,
                            answer=answer,
                            reasoning=reasoning,
                            python_code=python_code,
                            hint_level=st.session_state["hint_level"],
                        )
                    )
                except LearningServiceError as error:
                    st.error(str(error))
                else:
                    st.session_state["report"] = report
                    st.rerun()

with right:
    st.subheader("诊断报告")
    report_value = st.session_state.get("report")
    report = (
        DiagnosticReport.model_validate(report_value)
        if report_value is not None
        else None
    )
    if report is None:
        st.caption("提交答案后，这里会显示确定性诊断。")
        st.button("下一题", disabled=True, use_container_width=True)
    else:
        if report.overall_correctness == 1.0:
            st.success("正确")
        elif report.overall_correctness > 0:
            st.warning(f"部分证据：{report.overall_correctness:.2f}")
        else:
            st.error("暂未答对")

        st.markdown("**四维诊断**")
        for dimension, score in report.dimension_scores.model_dump().items():
            st.caption(f"{dimension}: {score:.2f}")
            st.progress(score)

        st.markdown("**诊断证据**")
        for evidence in report.evidence:
            st.write(f"- {evidence}")

        st.markdown("**可能误区**")
        if report.misconception_tags:
            st.write("、".join(report.misconception_tags))
        else:
            st.caption("当前没有确定的误区标签。")

        st.markdown("**反馈与下一步建议**")
        st.write(report.feedback)
        st.info(report.recommended_action)
        st.caption(report.uncertainty)

        next_disabled = report.next_question_id is None
        if st.button(
            "下一题",
            disabled=next_disabled,
            use_container_width=True,
        ):
            st.session_state["pending_question_id"] = report.next_question_id
            st.rerun()
