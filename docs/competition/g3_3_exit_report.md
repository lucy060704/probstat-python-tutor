# G3.3 本地产品主路径阶段门候选报告

## 1. 主 Agent 结论

截至 2026-08-08，G3.3 最终阶段门：**PASS**。独立评委 Agent 复审确认 P0=0、P1=0，允许
进入 G3.4。首审曾如实判定 HOLD；主 Agent 关闭两项 P1 后才获得 PASS：

1. Windows `--check` 原先可能因 CMD 括号内提前展开 `%ERRORLEVEL%` 而假通过；现改为运行时
   `if errorlevel` 失败分支，并有回归测试；
2. 教师小样本原先只隐藏率和均值、仍显示精确计数；现总体小于 3 人时所有精确计数为 `None`，
   逐知识点小样本行完全隐藏，Pydantic 也拒绝不一致 DTO。

本任务没有配置或调用 ADP，没有上传教材 PDF，没有新增生产依赖，也没有修改冻结的 G3.2
检索器、development/holdout 数据或评测器。

本报告只能证明本地工程候选版满足既定产品门槛，不等于真实学生学习效果，不等于教师已审核
15 张知识卡，也不构成“一等奖概率”的保证。

## 2. 验收矩阵

| G3.3 要求 | 实现与证据 | 主 Agent 结果 |
| --- | --- | --- |
| 学生完整主路径 | 随机匿名档案；选题、作答、答错自动 L1、四级提示、确定性诊断、本地引用、订正、推荐、下一题 | 通过 |
| 教师匿名汇总 | 只从 `learner_states.state_json` 读取 `LearningState`；DTO 无 ID/原始作答；总体与逐知识点均执行 `k>=3` | 通过 |
| macOS/Windows 一键启动 | `scripts/start_macos.command`、`scripts/start_windows.cmd`，均支持 `--check` | 通过 |
| 无 API Key | 页面明确显示离线可用；5 次主路径全部在无 Key 模式运行 | 通过 |
| 模型/网络失败降级 | 仅包围可选 `Runner.run` 与模型输出校验；失败后返回并提交确定性报告 | 通过 |
| 安全故障日志 | JSONL 只允许 6 类字段；不记录异常消息、身份、作答、代码、环境或密钥 | 通过 |
| 连续 5 次演示 | 5 个隔离匿名档案，每次均答错、核对 L1–L4、订正、诊断、推荐、进入下一题 | 5/5，通过 |
| 学习状态安全 | 模型失败仍原子提交；存储/判题/事务失败仍回滚并显示安全中文错误 | 通过 |
| Python 安全 | 页面与后端只做 AST 静态分析，不执行学习者代码 | 通过 |

## 3. 隐私边界

- 学生页面不提供姓名或学号输入，浏览器会话获得 `local_anon_<随机值>`；界面只显示后 8 位；
- `LearningState` 历史只包含题目、知识点、难度、分数、证据调整、正确性和提示级别；
- 教师汇总调用 `LearningStateStore.load_anonymized_states()`，SQL 不选择 `learner_id`，也不读取
  `submission_receipts.report_json`；
- 小于 3 个有作答档案时隐藏总体正确率；每个知识点也按独立有作答档案数执行 `k=3`；
- 本地 SQLite 未加密，正式试点前仍必须补充知情同意、数据保留期、访问控制和安全删除流程。

## 4. 故障恢复边界

`DiagnosticReport.delivery_mode` 区分四种交付状态：纯本地确定性、在线增强、模型失败回退和
安全隔离。模型异常或结构化输出不合法时，系统记录一个脱敏事件并使用已经生成的确定性报告；
分数、证据、掌握度和推荐不会由模型编造。RAG 索引/查询不可用时只失去知识上下文，不影响判题。

日志字段固定为：`event_id`、`timestamp_utc`、`component`、`code`、`exception_type`、
`recovery`。日志自身不可写时静默降级，不能反过来阻断学习。存储、确定性 grader 或提交事务的
异常不被伪装成模型降级，仍按原有原子回滚路径处理。

## 5. 五次连续产品旅程

执行命令：

```bash
.venv/bin/python scripts/g3_product_demo.py \
  --output docs/competition/g3_3_demo_result.json
```

结构化结果：

- `run_count=5`，`serious_fault_count=0`，`passed=true`；
- 5/5 均验证 L1、L2、L3、L4；
- 5/5 均识别明确错误，订正后正确，生成建议并能进入下一题；
- 每个隔离档案恰有 2 条真实尝试；
- 教师汇总为 `ready`，5 个匿名档案、10 次作答；序列化 DTO 无原始答案字段；
- 另一次注入的模型网络失败成功回退，生成 1 条脱敏故障事件；
- 临时数据库自动清理，没有污染正式本地学习状态。

机器可读证据：`docs/competition/g3_3_demo_result.json`。

## 6. 真实浏览器审计与截图

审计使用当前本地 Streamlit 服务器和同一浏览器会话完成，不以单元测试代替可见产品检查。

基线发现：旧页面使用可编辑共享 `demo` ID、没有教师页、答错不会自动给 L1、报告不展示知识
引用、模型失败会中断提交，且没有启动器或故障日志。修复后逐步验证：

1. `05-after-student-start.jpg`：随机匿名档案、离线状态和三栏主路径；
2. `06-after-wrong-auto-hint.jpg`：明确答错后自动显示一级概念提示；
3. `07-after-correction-citations.jpg`：订正、四维诊断、推荐和 3 条本地引用；
4. `08-after-next-question.jpg`：下一题切换成功，输入与报告清空；
5. `09-after-teacher-aggregate.jpg`：教师匿名档案数、作答数、正确率和分知识点汇总；
6. `10-final-student-start.jpg`、`11-final-correction-citations.jpg`：最终中文能力标签和推荐文案复验。

基线与修复后图片均位于 `docs/competition/assets/g3_3/`。截图是审计证据的一部分，最终验收仍以
浏览器交互、自动化测试、五次演示和独立评委审查的组合为准。

## 7. 工程验证

- `ruff check .`：通过；
- `pytest -q --tb=short`：617/617 通过，用时 65.46 秒；
- G3.3 聚焦产品/存储/服务/界面测试：39/39 通过；
- 后续中文能力标签修复后的策略/智能体/服务/界面/演示回归：41/41 通过；
- macOS 启动器真实 `--check`：通过；
- Windows 启动器在 macOS 上完成静态路径、引号和预检分支检查，仍需在 Windows 实机双击复核；
- 五次结构化产品演示：5/5，无严重故障；
- 当前本地 Streamlit 预览已通过真实浏览器学生/教师双页交互。
- 独立评委首审：HOLD，P0=0、P1=2；修复后复审 49/49，最终 PASS，P0=0、P1=0；
- 评委提出的唯一过期类注释 P2 已在封存前同步修正。

## 8. 主要变更文件

- `app.py`
- `src/probstat_tutor/analytics.py`
- `src/probstat_tutor/observability.py`
- `src/probstat_tutor/schemas.py`
- `src/probstat_tutor/storage.py`
- `src/probstat_tutor/tutor_agent.py`
- `src/probstat_tutor/service.py`
- `scripts/start_macos.command`
- `scripts/start_windows.cmd`
- `scripts/g3_product_demo.py`
- `tests/test_app.py`
- `tests/test_teacher_dashboard.py`
- `tests/test_observability.py`
- `tests/test_launchers.py`
- `tests/test_g3_product_demo.py`

## 9. 剩余风险与下一项

1. Windows 启动器当前只有静态自动检查，仍需队友在 Windows 11 实机双击一次；这不阻塞 macOS
   本地候选版，但正式材料必须如实写明。
2. SQLite 为单机未加密存储，教师汇总不是账号隔离的生产后台；正式试点需补隐私治理。
3. 教师页面的汇总只是工程数据，不代表真实学生试点或教学提升证据。
4. 15 张知识卡继续保持 `pending_teacher_review`，不能宣称教师审核完成。
5. G3.3 阶段门通过后，下一项应是 G3.4：平台无关最小 HTTPS API 契约与健康检查；仍不配置 ADP。
