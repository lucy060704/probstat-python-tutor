# G3 本地工程总阶段门发布候选报告

## 1. 主 Agent 候选结论

截至 2026-08-08，主 Agent 对 G3.1–G3.6 的需求级审计结论为：**本地工程发布候选 PASS；只有
独立评委对最终同一哈希快照复审通过后，才能关闭阶段门**。

本结论证明 G1–G3 目标中的本地 RAG、完整学生/教师体验、平台无关 API、可靠性和评测可以在
当前仓库复现。它不证明教师已经批准内容、真实学生获得学习成效、Windows 11 已实机通过、比赛
材料已经制作、ADP 已适配或作品拥有任何确定获奖概率。

本轮没有配置、调用或发布 ADP，没有访问真实模型或外部网络，没有上传教材 PDF，没有执行学习
者 Python 文本，没有运行失效 blind/holdout 评测，也没有修改冻结 G3.2 检索器和评测器。

## 2. G3 需求级阶段矩阵

| 阶段 | 宽要求 | 权威证据 | 本轮重放 | 主 Agent |
| --- | --- | --- | --- | --- |
| G3.1 | 可信本地 RAG、引用、分级披露、无结果降级 | `g3_1_exit_report.md`、索引与专项测试 | 15 来源、478 切片、指纹一致 | PASS |
| G3.2 | 冻结检索评测、严格来源—章节召回、引用完整性 | `g3_2_exit_report.md`、冻结文件 | 只重跑 64 条 development，21 项门禁全过 | PASS |
| G3.3 | 学生完整闭环、教师匿名汇总、模型故障降级、页面证据 | `g3_3_exit_report.md`、JSON、11 张截图 | 5/5 隔离旅程，严重故障 0 | PASS |
| G3.4 | 平台无关 API、Pydantic、OpenAPI、错误/幂等路径 | `g3_4_exit_report.md`、API JSON、OpenAPI | 进程内 ASGI `passed=true` | PASS |
| G3.5 | 原子回滚、冲突、重试、超时、熔断、性能基线 | `g3_5_exit_report.md`、可靠性 JSON | 24/24 隔离诊断，P95 2.010 ms | PASS |
| G3.6 | 总审计、交付清单、限制、公开复现、终审 | 审计 JSON、重建 JSON、追踪表、手册、清单、本报告 | 13/13 机器检查；658 项公开测试；最终同哈希复审待完成 | 候选 PASS |

## 3. 可重复机器审计

执行：

```bash
.venv/bin/python scripts/g3_release_audit.py \
  --output docs/competition/g3_release_audit_result.json
```

当前输出：

- `engineering_gate_passed=true`；
- `competition_submission_ready=false`；
- 8 个深度单元、33 道题、33 个知识节点、46 条边和 22 章目录映射；
- 15 个原创 RAG 来源、478 个切片；
- 13 项需求/安全/证据检查全部 PASS；
- 交付清单逐文件记录相对路径、字节数和 SHA-256；
- 清单自身有稳定 SHA-256；审计结果、重建结果和评委结论作为自校验元数据，不自我引用；
- ADP 平台探针、内部执行计划、教材 PDF、SQLite、日志、缓存、private、blind/holdout 均不进入
  G1–G3 交付允许列表。

审计器使用 Pydantic 定义输出；通过领域加载器重建课程图谱和 RAG，而不是只搜索文件名。运行时
AST 扫描明确拒绝直接 `eval`、`exec`、`compile` 和 `os.system` 调用；交付文本扫描私钥、
OpenAI 风格密钥与腾讯 SecretId。扫描只能降低意外泄露风险，正式提交前仍需人工逐文件复核。

## 4. 本轮公开重放结果

所有命令均在 macOS、Python 3.11.15 下从项目根目录执行：

- `scripts/g1_offline_demo.py`：5/5 旅程通过；
- `scripts/g3_local_rag_demo.py`：返回精确引用，15 来源、478 切片；
- `python -m evals.rag_eval --split development`：64 条案例，严格 Recall@3 56/56，query-only
  8/8，引用正确 164/165，技术完整 165/165，无结果 8/8，披露违规 0/64，21 项门禁全过；
- `scripts/g3_product_demo.py`：5/5 学生旅程，教师档案 5、作答 10，原始答案字段未出现，模型
  故障降级通过；
- `scripts/g3_api_contract_demo.py`：健康、推荐、L1/L4、诊断、幂等、422/503 全部通过；
- `scripts/g3_reliability_demo.py`：同文重放 200、改文 409、回滚 503 且历史 0、有限重试成功、
  模型熔断降级、24/24 性能样本通过，P50 1.604 ms、P95 2.010 ms、最大 2.141 ms；
- `scripts/start_macos.command --check`：通过；
- OpenAPI 导出核对：通过；
- `pip check`：无损坏依赖；
- `ruff check .`：通过；
- `git diff --check`：通过；
- `scripts/run_release_tests.py`：658/658 公开测试通过；
- `scripts/g3_verify_rebuilt_release.py`：只按允许列表复制 163 个文件，0 个 blind/holdout；导入探针
  确认使用临时 `src`，G1、RAG、公开 development、产品、API、可靠性、macOS、OpenAPI、
  658 项公开测试、ruff、依赖、包内审计和逐字段快照比较共 14 个命令全部通过。

development 延迟和可靠性延迟只是当前机器的回归数据，不代表公网、ADP、真实模型或多人课堂
SLA。原仓库全量 pytest 可检查旧 blind 文件的隔离规则；公开交付套件明确排除直接依赖旧 blind
标签的历史测试。本轮没有执行 blind/holdout 评分或依据其标签调参。

## 5. 冻结与安全边界

- `src/probstat_tutor/rag/retrieval.py` SHA-256：
  `5e0d2f01c151204cdfe7d3e31c3754f201ae6f9fae4d550c8431e8dca9805371`；
- `evals/rag_eval.py` SHA-256：
  `9a59205a1a1adf7ae2fbf29e1790ab8acc46ea24d7e0f20113fbea07f0d5cd7a`；
- 模型名继续只从 `OPENAI_MODEL` 环境变量读取；无 Key 时离线核心可用；
- 确定性判题决定正确性和分数，模型只能解释，不能发明分数；
- 学习者代码只做有界 AST 静态分析，不编译、导入或执行；
- 学习状态和提交回执同一 SQLite 事务提交；相同键改正文返回冲突；
- 教师端只做匿名聚合并执行 `k=3` 隐藏，不暴露原始作答字段；
- API 当前只适合本机，不具备公网 TLS、鉴权、限流或生产密钥管理。

## 6. 新增与更新文件

- `scripts/g3_release_audit.py`：离线需求级审计与 SHA-256 允许列表；
- `scripts/run_release_tests.py`：无 blind/holdout 依赖的公开 pytest 入口；
- `scripts/g3_verify_rebuilt_release.py`：允许列表临时重建和逐命令验证；
- `tests/test_g3_release_audit.py`：审计范围、冻结哈希、隐私排除和诚实状态回归；
- `docs/competition/g3_release_audit_result.json`：机器审计与文件校验清单；
- `docs/competition/g3_rebuilt_release_result.json`：临时重建的脱敏机器结果；
- `docs/competition/g3_requirements_traceability.md`：比赛要求与 G1–G3 证据对应；
- `docs/competition/g3_local_demo_runbook.md`：macOS/Windows 复现顺序和故障处理；
- `docs/competition/g3_delivery_checklist.md`：允许/禁止内容与人工作业；
- `docs/competition/g3_exit_report.md`：本报告；
- `README.md`：G3.6 审计入口和状态边界；
- G3.3、G3.4、G3.5 三个 JSON：由公开演示重新生成。

未新增生产依赖。

## 7. 独立评委审计历史与最终放行规则

独立评委没有直接接受主 Agent 摘要：

1. 首审发现发布包排除了 blind 数据却仍包含直接加载该文件的历史测试，判为 P1/HOLD；
2. 主 Agent 排除该仓库专用测试，新增公开测试入口、仅允许列表复制回归和完整临时重建；
3. 定向复审确认结构性 P1 已修复，但发现说明文档和脱敏脚本在重建后又发生变化，旧 manifest
   不再代表当前树，因此再次 HOLD；
4. 最终放行必须严格按“先冻结全部允许列表文件 → 生成审计 → 用该 manifest 重建 → 再生成临时
   审计逐字比较 → 评委核对相同 manifest”的顺序完成。

评委最终 PASS/HOLD、P0/P1/P2 和验证命令写入
`docs/competition/g3_judge_verdict.md`。该文件与两个机器结果都属于自校验元数据，不进入其所
描述的源码清单，避免为了记录结论再次改变被审快照。主 Agent不会在评委 PASS 前关闭阶段门。

## 8. 仍然存在且不应伪造的缺口

1. 八个单元及 33 道题仍是 `pending_teacher_review`；教师签名缺失；
2. 用户已口头说明教师允许教材使用，但仓库内仍缺可归档许可证明；
3. 尚无 10–15 名真实学生的知情同意、匿名试点、任务完成率和问卷；
4. Windows 启动器只有静态测试，未在 Windows 11 实机重放；
5. 申报表、比赛版设计说明书和不超过 5 分钟的视频尚未完成；
6. 尚未进行 ADP 适配、正式发布或最终提交；
7. 因以上缺口，当前不能声称比赛提交就绪、内部评分达到 90/100 或有 90% 获奖概率。
