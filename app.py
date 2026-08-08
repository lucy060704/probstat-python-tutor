"""Local Streamlit product; learning and aggregation logic stays in services."""

import asyncio
import uuid

import streamlit as st

from probstat_tutor.analytics import TeacherDashboardStatus
from probstat_tutor.config import get_settings
from probstat_tutor.schemas import (
    CAPABILITY_LABELS_ZH,
    CapabilityDimension,
    ConceptId,
    DeliveryMode,
    DiagnosticReport,
    EvidenceVerdict,
    QuestionType,
)
from probstat_tutor.service import LearningService, LearningServiceError

CONCEPT_LABELS = {
    ConceptId.DATA_QUALITY: "数据加载、缺失值与数据质量",
    ConceptId.MEAN_MEDIAN: "均值与中位数",
    ConceptId.VARIANCE_STD: "方差与标准差",
    ConceptId.SAMPLING_STANDARD_ERROR: "抽样与标准误",
    ConceptId.CONFIDENCE_INTERVAL: "置信区间",
    ConceptId.PROBABILITY_SIMULATION: "概率规则与 NumPy 随机模拟",
    ConceptId.COMMON_DISTRIBUTIONS: "常用分布及其 Python 表达",
    ConceptId.JOINT_CORRELATION: "联合分布、相关与分组分析",
    ConceptId.HYPOTHESIS_TESTING: "假设检验与小型 A/B 数据分析",
}

QUESTION_TYPE_LABELS = {
    QuestionType.CONCEPT: "概念理解",
    QuestionType.PYTHON: "Python 实现",
    QuestionType.INTERPRETATION: "数据解释",
}


@st.cache_resource
def get_learning_service() -> LearningService:
    """Reuse curriculum, verified RAG index, and local storage across reruns."""

    return LearningService()


def reset_student_view() -> None:
    """Clear presentation state without placing business logic in the page."""

    for key in (
        "selection_signature",
        "current_question_id",
        "report",
        "answer",
        "reasoning",
        "python_code",
        "pending_question_id",
    ):
        st.session_state.pop(key, None)
    st.session_state["hint_level"] = 0


settings = get_settings()
service = get_learning_service()

st.set_page_config(page_title=settings.app_title, layout="wide")
st.title(settings.app_title)
st.caption("本地产品候选版 · 确定性判题 · 渐进提示 · 可追溯知识依据")

if service.offline_mode:
    st.info("离线可用：当前未配置 API Key，判题、提示、掌握度、知识检索和推荐仍可完整运行。")

st.session_state.setdefault("session_id", f"streamlit-{uuid.uuid4().hex}")
st.session_state.setdefault(
    "anonymous_learner_id", service.create_anonymous_learner_id()
)
st.session_state.setdefault("hint_level", 0)

learner_id = st.session_state["anonymous_learner_id"]
pending_question_id = st.session_state.pop("pending_question_id", None)
if pending_question_id:
    pending_question = service.get_question(pending_question_id)
    st.session_state["selected_concept"] = pending_question.concept_id
    st.session_state["current_question_id"] = pending_question.id
    st.session_state["selection_signature"] = (learner_id, pending_question.concept_id)
    st.session_state["hint_level"] = 0
    st.session_state["report"] = None
    for input_key in ("answer", "reasoning", "python_code"):
        st.session_state[input_key] = ""

student_tab, teacher_tab = st.tabs(["学生学习", "教师匿名汇总"])

with student_tab:
    left, middle, right = st.columns([1.05, 1.9, 1.25], gap="large")

    with left:
        st.subheader("学习状态")
        st.caption("本次使用随机匿名档案，不要输入姓名、学号或联系方式。")
        st.code(f"匿名档案 ···{learner_id[-8:]}", language=None)

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
            label = CAPABILITY_LABELS_ZH[CapabilityDimension(dimension)]
            st.caption(f"{label}：{score:.2f}")
            st.progress(score)

        st.markdown("**最近学习记录**")
        if dashboard.recent_records:
            st.dataframe(
                list(dashboard.recent_records),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("还没有学习记录。")

        if st.button("清空当前匿名档案", type="secondary", width="stretch"):
            service.reset_demo_learner(learner_id)
            reset_student_view()
            st.rerun()

    current_question_id = st.session_state.get("current_question_id")
    question = service.get_question(current_question_id) if current_question_id else None

    with middle:
        st.subheader("当前题目")
        if question is None:
            st.info("当前知识点暂时没有可作答题目，请查看左侧提示或选择其他知识点。")
        else:
            st.markdown(f"### {question.title}")
            st.caption(
                f"题型：{QUESTION_TYPE_LABELS[question.question_type]} · "
                f"难度系数：{question.difficulty:.2f}"
            )
            st.write(question.prompt)
            with st.expander("查看数据预览", expanded=True):
                st.json(question.dataset, expanded=True)

            answer = st.text_input("答案输入", key="answer", placeholder="请输入答案")
            reasoning = st.text_area(
                "思考过程输入",
                key="reasoning",
                placeholder="请写出你的判断依据或计算步骤",
                height=110,
            )
            python_code = st.text_area(
                "Python 代码输入（仅作静态文本分析，绝不执行）",
                key="python_code",
                placeholder="可选：粘贴你的 Python 代码",
                height=120,
            )

            hint_column, submit_column = st.columns(2)
            with hint_column:
                if st.button("请求提示", width="stretch"):
                    st.session_state["hint_level"] = min(
                        4, st.session_state["hint_level"] + 1
                    )
            with submit_column:
                submitted = st.button("提交", type="primary", width="stretch")

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
                        if report.overall_correctness < 1.0:
                            st.session_state["hint_level"] = max(
                                1, st.session_state["hint_level"]
                            )
                        st.rerun()

    with right:
        st.subheader("诊断与下一步")
        report_value = st.session_state.get("report")
        report = (
            DiagnosticReport.model_validate(report_value)
            if report_value is not None
            else None
        )
        if report is None:
            st.caption("提交答案后，这里会显示可追溯的确定性诊断。")
            st.button("下一题", disabled=True, width="stretch")
        else:
            if report.overall_correctness == 1.0:
                st.success("回答正确")
            elif report.overall_correctness > 0:
                st.warning(f"部分证据成立：{report.overall_correctness:.2f}")
            else:
                st.error("暂未答对，已自动开启一级提示")

            if report.delivery_mode == DeliveryMode.MODEL_FALLBACK:
                st.warning(report.delivery_message_zh)
            elif report.delivery_mode == DeliveryMode.SAFETY_ISOLATED:
                st.info(report.delivery_message_zh)
            else:
                st.caption(report.delivery_message_zh)

            st.markdown("**四维诊断**")
            for dimension, score in report.dimension_scores.model_dump().items():
                label = CAPABILITY_LABELS_ZH[CapabilityDimension(dimension)]
                st.caption(f"{label}：{score:.2f}")
                st.progress(score)

            with st.expander("查看诊断证据"):
                for evidence in report.evidence:
                    st.write(f"- {evidence}")

            st.markdown("**可能误区**")
            blocking_findings = [
                finding
                for finding in report.grader_findings
                if finding.verdict != EvidenceVerdict.SUPPORTS
            ]
            if blocking_findings:
                for finding in blocking_findings:
                    st.write(f"- {finding.message_zh}")
            elif report.misconception_tags:
                st.caption("已记录可核查的误区标签，请结合诊断证据订正。")
            else:
                st.caption("当前没有确定的误区标签。")

            st.markdown("**反馈与下一步建议**")
            st.write(report.feedback)
            st.info(report.recommended_action)
            st.caption(report.uncertainty)

            with st.expander("本地知识依据与引用"):
                st.write(report.knowledge_context_message)
                if report.knowledge_citations:
                    for citation in report.knowledge_citations:
                        st.markdown(
                            f"**[{citation.citation_id}] {citation.source_title}**  "
                            f"章节：`{citation.section}` · 内容版本：`{citation.source_version}`"
                        )
                        if citation.quote:
                            st.caption(citation.quote)
                else:
                    st.caption("本次没有可展示的匹配引用，系统未编造来源。")

            if st.button(
                "下一题",
                disabled=report.next_question_id is None,
                width="stretch",
            ):
                st.session_state["pending_question_id"] = report.next_question_id
                st.rerun()

with teacher_tab:
    st.subheader("班级匿名学习概览")
    teacher_dashboard = service.get_teacher_dashboard()
    st.info(teacher_dashboard.privacy_notice_zh)

    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "匿名档案数",
        teacher_dashboard.profile_count
        if teacher_dashboard.profile_count is not None
        else f"少于 {teacher_dashboard.minimum_cohort_size}",
    )
    metric_columns[1].metric(
        "有作答档案",
        teacher_dashboard.attempted_profile_count
        if teacher_dashboard.attempted_profile_count is not None
        else f"少于 {teacher_dashboard.minimum_cohort_size}",
    )
    metric_columns[2].metric(
        "作答次数",
        teacher_dashboard.total_attempt_count
        if teacher_dashboard.total_attempt_count is not None
        else "已隐藏",
    )

    if teacher_dashboard.status == TeacherDashboardStatus.NO_DATA:
        st.caption("还没有匿名学习记录。学生完成作答后，这里会自动形成汇总。")
    elif teacher_dashboard.status == TeacherDashboardStatus.SUPPRESSED:
        st.warning(
            f"当前不足 {teacher_dashboard.minimum_cohort_size} 个有作答匿名档案，正确率已隐藏。"
        )
    else:
        st.metric(
            "总体正确率",
            f"{teacher_dashboard.overall_correct_rate:.1%}"
            if teacher_dashboard.overall_correct_rate is not None
            else "已隐藏",
        )

    concept_rows = []
    for summary in teacher_dashboard.concept_summaries:
        if summary.suppressed:
            continue
        concept_rows.append(
            {
                "知识点": CONCEPT_LABELS[summary.concept_id],
                "有作答档案": summary.attempted_profile_count,
                "作答次数": summary.attempt_count,
                "正确率": (
                    f"{summary.correct_rate:.1%}"
                    if summary.correct_rate is not None
                    else "已隐藏"
                ),
                "平均提示级别": (
                    f"{summary.average_hint_level:.2f}"
                    if summary.average_hint_level is not None
                    else "已隐藏"
                ),
            }
        )
    if concept_rows:
        st.dataframe(concept_rows, hide_index=True, width="stretch")
    else:
        st.caption("暂无可显示的知识点汇总；少于 3 人的知识点行已隐藏。")

    with st.expander("本地数据与故障恢复说明"):
        st.write("学习状态只保存在当前设备；教师页不读取包含原始作答的提交回执。")
        st.write("在线模型或网络不可用时会自动回退到确定性诊断，并写入不含身份和答案的故障事件。")
        st.caption("正式试点前仍需补充知情同意、数据保留期与设备访问控制。")
