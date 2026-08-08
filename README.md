# 概率统计 × Python 数据分析学习诊断智能体

这是一个面向初学者的简体中文学习与诊断应用。当前已完成可离线运行的单教学智能体、
确定性多证据判题、四维掌握度、推荐和 SQLite 幂等学习流程。题库现有 33 题：原有
4 个统计知识点 12 题，加上 G2 已完成工程实现的 8 个深度单元。其中描述统计、抽样推断和
参数估计复用并升级原题，数据质量、概率模拟、常用分布和联合分布与相关各新增 3 题；抽样
推断和参数估计与置信区间各另增 2 道深度题，假设检验/A-B 新增 5 道综合题和一份纯合成数据。

## 环境要求

- Python 3.11 或更高版本
- 建议使用项目根目录中的 `.venv`

## 安装

在 PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 本地运行

完成安装后，可以直接双击：

- macOS：`scripts/start_macos.command`
- Windows：`scripts/start_windows.cmd`

两个启动器都会先确认项目自带虚拟环境存在，再只监听本机 `127.0.0.1`。不启动网页的预检命令：

```powershell
.\scripts\start_windows.cmd --check
```

macOS 对应命令为 `./scripts/start_macos.command --check`。也可以继续使用解释器直接启动：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

启动成功后，终端会显示本地访问地址。页面包括完整学生学习闭环和教师匿名汇总。

## 质量检查

```powershell
.\.venv\Scripts\python.exe scripts\run_release_tests.py
.\.venv\Scripts\python.exe -m ruff check .
```

公开测试运行器明确排除仓库内依赖已失效 blind 标签的历史隔离测试，并在启动前拒绝任何实际
blind 文件路径依赖。维护原工作树时仍可额外运行 `python -m pytest`；它会验证旧数据的隔离
约束，但不是公开交付包的复现命令，也不能用于调参。

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
- `data/questions.yaml`：9 个可练习知识点、三种题型组成的 33 道内置题；
- `data/curriculum_catalog.yaml`：8 个深度单元、知识节点关系和 8+14 章目录映射；
- `data/rag/sources/`：团队原创、带版本和 checksum 的课程知识卡；
- `src/probstat_tutor/rag/retrieval.py`：离线切片、索引、检索、引用和无结果降级；
- `tests/`：pytest 测试；
- `docs/product_spec.md`：v0.1 产品规格与验收基线。

## 验证题库

以下命令会读取 YAML，并用 Pydantic 检查字段、难度、权重、题型和前置知识点：

```powershell
.\.venv\Scripts\python.exe -c "from probstat_tutor.curriculum import load_default_question_bank; print(len(load_default_question_bank().questions))"
```

成功时输出 `33`。题库文件不存在、YAML 损坏或字段不符合规则时，加载器会输出明确的中文错误。

课程图谱只把两本书的 22 章作为目录映射；`coverage_level` 的机器语义固定为
`target_only_not_implementation_status`，不表示相应功能已经完成。下面的命令会同时
检查 8 个深度单元、知识节点前置图、22 章映射以及已进入教师待审状态的单元题目：

```powershell
.\.venv\Scripts\python.exe -c "from probstat_tutor.curriculum_graph import load_default_curriculum_catalog; print(len(load_default_curriculum_catalog().units))"
```

当前 8 个深度单元全部进入 `pending_teacher_review`；它们均未获得教师批准，
也没有任何内容被标记为教师已审核。

## 本地原创知识库检索

G3.1 已把 manifest 登记的 15 张团队原创 YAML 知识卡建立为可确定性重建的本地索引。
索引只接受 `project_authored + project-owned + low`、同时允许检索和引用的资料；教材 PDF、
题目答案、rubric 和评测标签不会进入索引。每个命中都返回资料 ID、版本、切片 ID、两级
checksum 和精确摘录。没有匹配或索引不可用时返回固定降级说明，不编造来源，也不阻断判题。

检索还遵守四级提示边界：一级只提供学习目标、前置知识和反思问题；二级可提供概念解释、
一般误区、数据解释原则和不含表达式的公式含义；三级才允许公式表达式和 Python API；
四级才允许总结。它作为服务端确定性上下文接入诊断报告，
没有新增第六个模型工具，模型不能更改引用。

```powershell
.\.venv\Scripts\python.exe scripts\g3_local_rag_demo.py "均值 中位数 异常值" --concept mean_median
```

macOS 或 Linux 可将命令中的解释器替换为 `.venv/bin/python`。此演示不需要 API Key 或网络。

G3.2 另建立了不含正式题答案的本地检索评测。64 条冻结 development 案例包含
56 个正例和 8 个域外负例，每单元有 1 个不带 concept/node 的 L4 综合问法和 1 个
真实 L1 泄漏保护案例。正式指标使用“来源—章节对” Recall@3，不用更宽松的来源
命中代替；每个引用还必须通过版本、双 checksum、精确摘录、相关章节和披露等级核对。

```powershell
.\.venv\Scripts\python.exe -m evals.rag_eval
```

运行前会强制校验数据集、索引、检索器、评测器以及 15 个来源版本的冻结指纹。
当前 development 基线为：严格 Recall@3 `56/56`、query-only Recall@3 `8/8`、引用相关正确
`164/165`、引用技术完整 `165/165`、no-match `8/8`、披露违规 `0/64`、三次重放稳定
`64/64`。最终全新独立 holdout 严格 Recall@3 `40/40`、引用正确 `117/120`、引用技术完整
`120/120`、no-match `8/8`，21 项门禁全部通过。首轮 HOLD、通用修复和两次运行的完整诚实
记录见 `docs/competition/g3_2_exit_report.md`；合同与隔离流程见 `docs/rag_eval_spec.md`。

## 确定性判题器

`src/probstat_tutor/graders.py` 提供数值、选择题、文字证据、DataFrame 结果和多证据组合判题。
答案、思考过程和 Python 文本会分别形成带规则 ID 的确定性 finding；答案正确但明确推理或代码
错误时，整体仍判为错误。Python 只通过有长度、节点数和深度限制的 AST 提取结构特征，并从
最后结果向前追踪简单变量赋值，检查根节点、运算符和所需调用是否连接；不调用大模型，不使用
`eval()`、`exec()` 或 `compile()`，也不会导入或执行学习者代码。

结构不匹配时，题库可以继续用 AST 可观察特征区分“方法只引用未调用”“只取一个中间位置”
“把方差当标准差”或“NumPy 标准差缺少 `ddof=1`”等具体根因；无法可靠细分时才返回通用
结构冲突。推荐文案由主要 finding 的规则 ID、判定类型和能力维度确定性生成。答错、证据不足、
无关或不安全时只引导重试当前题，不同时暴露下一题 ID。

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
掌握度、推荐类别、推荐依据和下一题。

模型名只从 `OPENAI_MODEL` 读取，不在代码中写死。当前诊断关键路径不写 Agents SDK 会话；
学习状态和提交回执在同一个 SQLite 事务中保存，避免模型或提交失败时只写入一半事实。
数据库文件位于 `data/`，已被 `.gitignore` 排除。在线模型或输出校验失败时，系统不会放弃本次
确定性判题：它会自动返回本地诊断并原子保存学习状态。`data/logs/faults.jsonl` 只记录随机事件
编号、UTC 时间、组件、稳定错误码、异常类型和恢复动作，不记录异常消息、匿名 ID、答案、代码、
环境变量或密钥。

没有 `OPENAI_API_KEY` 时自动进入离线演示模式：确定性判题、掌握度更新、下一题和结构化诊断
仍然运行，但不会调用 OpenAI API。自由文本在离线模式下只能做严格答案比较；证据不足时报告会
明确写“不确定”，并要求学习者补充推理。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_tutor_agent.py
```

## Streamlit 学习界面与教师匿名汇总

学生页分为学习状态、当前题目和诊断/下一步三栏，覆盖“选题—作答—分级提示—诊断—本地
引用—推荐—下一题”。首次答错自动开启一级概念提示，第四级才给完整解释。页面只负责收集输入、
调用 `LearningService` 和展示结果；判题、掌握度、幂等提交与推荐逻辑不在 `app.py` 中。

每个浏览器会话使用随机本地匿名档案，不要求姓名、学号或联系方式。教师页只读取
`LearningState` 的匿名状态，不读取包含原始作答摘录的提交回执；总体和每个知识点单元格都使用
`k=3` 小样本隐藏。它只是本机工程汇总，不是生产级班级管理系统；正式试点仍需知情同意、数据
保留期和设备访问控制。

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

学习状态和提交回执保存在 SQLite 中。相同内容被连续提交时直接返回已保存的诊断报告，
不会重复更新学习记录。“清空当前匿名档案”会清除当前档案的掌握度、历史和提交回执。

G3.3 产品验收脚本会在临时目录连续完成 5 个隔离匿名档案的答错、四级提示核对、订正、诊断、
推荐和下一题，并另做一次模型网络故障注入；不会发送真实网络请求，也不会污染本地学习数据库：

```powershell
.\.venv\Scripts\python.exe scripts\g3_product_demo.py --output docs\competition\g3_3_demo_result.json
```

## 平台无关 API 契约

G3.4–G3.5 提供一个不依赖赛事平台的 ASGI 适配层，包含 `/health`、`/v1/recommend`、`/v1/hint`
和 `/v1/diagnose`。路由只调用 `LearningService`；公开题目不包含答案或评分规则，所有 POST 请求
使用 Pydantic `schema_version`、`request_id` 和 `idempotency_key` 契约。

G3.5 为诊断回执保存请求指纹：相同幂等键和相同正文可安全重放，同一键更改正文会返回 409，
不会返回旧答案对应的报告。可选在线解释使用硬超时、最多 3 次的有限尝试和进程内熔断；模型失败
只影响自然语言增强，确定性判题与原子学习记录仍可用。未来客户端只对明确标记为可重试的
500/503 做最多 3 次尝试，POST 必须复用同一个幂等键；409/422 不自动重试。

## G1–G3 总验收与安全交付清单

G3.6 提供离线发布审计器，把 G1–G3 的课程、RAG、演示、冻结文件、安全边界和阶段证据一次
重建并生成 SHA-256 文件清单。清单采用允许列表，明确排除本地 SQLite、故障日志、教材 PDF、
缓存、密钥、private 资料和已经失效的 blind/holdout 数据：

```powershell
.\.venv\Scripts\python.exe scripts\g3_release_audit.py `
  --output docs\competition\g3_release_audit_result.json
.\.venv\Scripts\python.exe scripts\run_release_tests.py
.\.venv\Scripts\python.exe scripts\g3_verify_rebuilt_release.py `
  --output docs\competition\g3_rebuilt_release_result.json
```

最后一个命令只使用允许列表把源码复制到临时目录，强制从临时 `src` 导入，并重跑全部公开
演示、658 项公开测试和质量门；不会复制原工作树、blind/holdout 或本地状态。macOS 可把解释器
换为 `.venv/bin/python`。机器审计通过时会得到
`engineering_gate_passed=true`；在教师签字、真实学生试点、Windows 11 实机、申报材料、视频和
赛事平台适配完成前，`competition_submission_ready` 会故意保持 `false`。完整复现顺序见
`docs/competition/g3_local_demo_runbook.md`，人工作业见
`docs/competition/g3_delivery_checklist.md`。

本地服务器固定监听 `127.0.0.1`，当前不能直接暴露公网。未来如果获准部署，必须在受控 TLS
终止层后只提供 HTTPS，并另加鉴权、限流和密钥管理。完整合同见 `docs/api_contract.md`，机器可读
OpenAPI 见 `docs/api/openapi.json`。

```powershell
.\.venv\Scripts\python.exe scripts\run_local_api.py --check
.\.venv\Scripts\python.exe scripts\g3_api_contract_demo.py --output docs\competition\g3_4_api_contract_result.json
.\.venv\Scripts\python.exe scripts\g3_reliability_demo.py --output docs\competition\g3_5_reliability_result.json
.\.venv\Scripts\python.exe scripts\export_api_contract.py --check
```

可靠性参数及本地 P95 复现方法见 `docs/reliability.md`。该结果只代表记录的本机、离线 ASGI
内存传输和合成故障，不代表公网、赛事平台或真实模型 SLA。

## 诊断评测

`evals/cases.jsonl` 保存人工标注的学习者回答。评测强制使用离线模式，不调用真实 API，
并分别报告确定性判题、误区标签、下一步建议、一级提示泄露、延迟和 API 失败情况。
这些指标不会合并成一个含义模糊的总分。

```powershell
.\.venv\Scripts\python.exe evals\run_evals.py
```

G1 离线阶段门还提供五次连续学习旅程脚本。每次都验证一级提示、错误诊断、订正、下一题和重复
提交幂等，不需要 API Key 或网络：

```powershell
.\.venv\Scripts\python.exe scripts\g1_offline_demo.py
```

教师抽查时先发送 `docs/competition/g1_teacher_review_form.md`，教师独立填写后再查看
`docs/competition/g1_teacher_review_system_key.md`，避免系统答案影响人工判断。
