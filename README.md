# 概率统计 × Python 数据分析学习诊断智能体

这是一个面向初学者的简体中文学习与诊断应用。目前已经建立项目骨架，并完成
经过 Pydantic 验证的 v0.1 十二题题库、不依赖大模型的确定性判题器，以及启发式掌握度
与下一题策略；智能体教学逻辑尚未实现。

## 环境要求

- Python 3.11 或更高版本
- 建议使用项目根目录中的 `.venv`

## 安装

在 PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 本地运行

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

启动成功后，终端会显示本地访问地址。当前页面只用于验证项目和配置可以正常加载。

## 质量检查

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

## OpenAI 配置

运行骨架页面不需要 API Key。后续启用模型功能时，请通过环境变量配置，切勿把真实密钥写入文件：

```powershell
$env:OPENAI_API_KEY = "你的 API Key"
$env:OPENAI_MODEL = "你的模型名称"
```

`.env.example` 只展示变量名，不应保存真实密钥。

## 当前目录职责

- `app.py`：最小 Streamlit 页面，不放业务逻辑；
- `src/probstat_tutor/`：后续业务逻辑所在的 Python 包；
- `data/questions.yaml`：四个知识点、三种题型组成的 12 道内置题；
- `tests/`：pytest 测试；
- `docs/product_spec.md`：v0.1 产品规格与验收基线。

## 验证题库

以下命令会读取 YAML，并用 Pydantic 检查字段、难度、权重、题型和前置知识点：

```powershell
.\.venv\Scripts\python.exe -c "from probstat_tutor.curriculum import load_default_question_bank; print(len(load_default_question_bank().questions))"
```

成功时输出 `12`。题库文件不存在、YAML 损坏或字段不符合规则时，加载器会输出明确的中文错误。

## 确定性判题器

`src/probstat_tutor/graders.py` 提供数值、选择题、文字关键词辅助证据和 DataFrame
结果判题。它们只比较数据，不调用大模型，也不会执行学习者提交的 Python 代码。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_graders.py
```

## 掌握度与下一题策略

学习状态按“知识点 × `concept` / `calculation` / `python` / `interpretation`”保存。
每次作答先按提示层级降低证据置信度，再根据题目的维度权重更新相关单元格：

```text
new = clip(0.7 × old + 0.3 × weighted_adjusted_evidence, 0, 1)
```

前置知识点是否达标使用四维掌握度的平均值，阈值为 `0.60`。推荐策略还会考虑最弱维度、
当前掌握度、题目难度、最近三题以及连续成功或失败。

**重要说明：这只是 v0.1 的启发式模型，不是经过教育测量验证的模型。** 分数只能用于
本题库内的学习引导，不能用于考试评价、能力认证或高风险决策。

## 单教学智能体

`TutorAgent` 使用 OpenAI Agents SDK 的单个 `Agent`，没有 handoff 或其他智能体。它提供五个
本地工具：读取当前题、确定性判题、读取学习状态、更新学习状态和选择下一题。诊断结果使用
Pydantic `DiagnosticReport`，模型只能润色反馈，不能修改由工具锁定的分数、证据、误区标签、
掌握度和下一题。

模型名只从 `OPENAI_MODEL` 读取，不在代码中写死。会话由 Agents SDK `SQLiteSession` 保存，
学习状态保存在独立 SQLite 表中。数据库文件位于 `data/`，已被 `.gitignore` 排除。

没有 `OPENAI_API_KEY` 时自动进入离线演示模式：确定性判题、掌握度更新、下一题和结构化诊断
仍然运行，但不会调用 OpenAI API。自由文本在离线模式下只能做严格答案比较；证据不足时报告会
明确写“不确定”，并要求学习者补充推理。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_tutor_agent.py
```

## Streamlit 学习界面

单页 MVP 分为学习状态、当前题目和诊断报告三栏。页面只负责收集输入、调用
`LearningService` 和展示结果；判题、掌握度、幂等提交与推荐逻辑不在 `app.py` 中。

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

学习状态和提交回执保存在 SQLite 中。相同内容被连续提交时直接返回已保存的诊断报告，
不会重复更新学习记录。“重置演示学习者”会清除该 ID 的掌握度、历史和提交回执。

## 诊断评测

`evals/cases.jsonl` 保存人工标注的学习者回答。评测强制使用离线模式，不调用真实 API，
并分别报告确定性判题、误区标签、下一步建议、一级提示泄露、延迟和 API 失败情况。
这些指标不会合并成一个含义模糊的总分。

```powershell
.\.venv\Scripts\python.exe evals\run_evals.py
```
