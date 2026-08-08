# G3.2 本地 RAG 检索评测规范

## 1. 目的与边界

本规范回答四个问题：检索是否找到了正确来源的正确章节，引用是否真实可核验，
域外问题是否明确返回无结果，以及渐进式提示是否越级。评测完全在本地运行，不调用
模型、网络或赛事平台，不使用正式题答案、grader 标签或学习者数据。

这是检索工程评测，不等于知识内容已经教师审批，也不等于已证明真实教学效果。

## 2. 数据集合同

可见 development 集位于 `evals/rag/development.jsonl`，每行由 `RagEvalCase` 严格校验。
重要字段如下：

| 字段 | 含义 |
|---|---|
| `query_text` | 学习者问法，禁止携带 source/chunk ID、checksum 或文件路径 |
| `query_family` | 查询变体家族，development 与 holdout 不得重叠 |
| `metadata_mode` | `metadata_assisted` 使用 concept/node；`query_only` 只使用自然语言 |
| `purpose` | 实际调用目的：`diagnostic`、`hint` 或 `knowledge_search` |
| `gold_concept_ids` | query-only 正例的人工金标知识点 |
| `required_targets` | `any_of` 严格 `(source_id, section)` 目标，用于正式 Recall@k |
| `acceptable_targets` | 允许成为支持性引用的来源—章节对 |
| `disclosure_level` | 本次检索最高允许的提示级别 L1–L4 |
| `forbidden_sections/fragments` | 案例级禁止章节或片段，供安全对抗案例使用 |
| `origin` | development 必须是 `team_authored`，holdout 必须是 `independent_judge` |

`required_targets` 和 `acceptable_targets` 都将来源与章节绑定，不允许用两个平面列表产生错误
笛卡尔积。一个 matched 案例的 Top-3 至少要有一个严格目标；同时，所有返回引用都必须
属于该案例人工金标知识点、该级别可披露的章节，并通过引用完整性核对。

## 3. Development 分布

冻结集共 64 条：

- 8 个深度单元各 7 个 matched，共 56 个正例；
- 每单元新增 1 个 L4、query-only 综合复习问法，共 8 个；
- 8 个不带 concept/node 的域外 no-match 负例；
- 覆盖 9 个 `ConceptId`、15 张知识卡，每个来源至少 2 个严格正例；
- 披露级别同时覆盖 L1、L2、L3 和 L4。
- `diagnostic`、`hint`、`knowledge_search` 分别覆盖 37、11、16 条；
- 每单元有 1 个真实 L1 案例带禁止章节和普通中文结论片段，共 8 个。

冻结清单在 `evals/rag/development_manifest.json`，同时记录数据集、索引、检索器和
评测器四类指纹。Runner 在计算指标前会强制比较这些指纹以及 15 个来源的版本/checksum；
任何一项变更，都会先拒绝运行，旧报告不能继续当作当前证据。

## 4. 指标与硬门

### 4.1 严格目标 Recall

```text
Target Recall@3
= Top-3 至少命中一个 required (source, section) 的正例数
  / matched 案例总数
```

硬门：总体 Recall@3 不低于 90%，每单元不低于 80%，query-only 层不低于 75%。
同时报告 Recall@1、来源级 Recall@1/3 和 MRR，但更宽松的来源级指标不能代替正式指标。

### 4.2 引用正确率

每个引用必须同时满足：

1. citation ID 与排名一致；
2. chunk ID 存在，来源、版本、标题和章节与切片一致；
3. content/source checksum 一致；
4. 内容完全一致，摘录必须精确等于完整短文或规定的 599 字前缀加省略号；
5. `(source_id, section)` 属于本案例 `acceptable_targets`；
6. 引用元数据和正文推断的最低披露级别都不超限。

引用相关正确率门槛是 95%；其中 1–4 的技术完整性必须是 100%，不允许用其他高分
抵消伪造或 checksum 错误。

### 4.3 无结果、披露与稳定性

- no-match 准确率必须是 100%，且 hits/citations 为空；
- 全等级披露违规率和 L1 泄漏率必须是 0；
- 受保护 L1 案例至少 8 个且每单元至少 1 个，禁止片段规范化后不得为空；
- 每条案例连续运行 3 次，结果、排序、分数、引用和指纹必须完全一致；
- schema/runtime 失败率必须是 0；
- 延迟报告平均值和 p95，但因受本机负载影响，不作为质量否决门。

## 5. 防泄漏与防虚高

加载器会拒绝：

- 重复 ID、重复规范化查询或重复完整查询合同；
- 正式题完整题干或正式题内部 ID；
- 知识卡完整切片、高度近重复改写或完整包含关系；
- source ID、chunk ID、checksum 和资料路径；
- 与 unit/concept 不一致或在当前披露级别完全不可用的人工目标。

开发调整只允许基于可见 development 和独立评委返回的聚合失败类别。本轮通用调整包括：

- 为明确的“总结 / Python / 公式”请求保留强章节意图，为“误区 / 解释 / 前置 / 反思 /
  概念”使用较小且封顶的章节意图；多种意图可以同时生效；
- 有知识节点元数据且存在合规重叠候选时，过滤不重叠候选；没有合规重叠时回退到词法检索；
- 只有节点元数据唯一定位到一个来源时，允许该来源占满 Top-3；其他情况仍限制每来源 2 条；
- query-only 必须含一个中英文课程特异锚点，或至少两个一般统计锚点，并继续满足原词法分数门槛。

这些规则不修改 BM25 参数、全局 `top_k` 或基础阈值，也不包含案例 ID、正式答案或封存题特判。

## 6. 独立 holdout 程序

1. 先冻结数据集、索引、检索器和评测器指纹。
2. 评委 Agent 在仓库之外的临时目录撰写至少 48 条 `origin=independent_judge` 案例；
   每单元至少 5 个 matched，另有至少 8 个 no-match 和 8 个 query-only matched。
3. 案例在首次运行前冻结 SHA-256，然后用通用 CLI 一次运行。
   Runner 会先重新校验 development 冻结清单，再拒绝与 development 重复的 query family、
   规范化查询、完整查询合同或 4-gram 高近重复查询。
4. 主 Agent 只接收指纹、分布、聚合指标、失败类别计数和 PASS/HOLD；不接收查询、标签、
   失败 ID 或引用正文。
5. 如果冻结代码变更，先使旧报告失效，再记录重跑次数；不允许根据封存案例逐题特判。

当前 Agent Team 共享工作区，因此这种方式是“流程隔离的一次封存评测”，不是密码学意义的
真盲测。正式报奖前，建议再由教师、队友设备或 CI secret 环境复核一次。

## 7. 本地命令

```bash
.venv/bin/python -m evals.rag_eval
.venv/bin/pytest -q tests/test_g3_rag_eval.py tests/test_g3_local_rag.py
```

封存评委使用：

```bash
.venv/bin/python -m evals.rag_eval \
  --dataset /outside/repository/holdout.jsonl \
  --split holdout \
  --freeze-manifest /outside/repository/holdout_manifest.json
```

CLI 成功时只输出聚合 `RagEvalSummary`，不输出查询、标签或完整检索正文。

## 8. 当前冻结阶段门

最终冻结版本的 development Target Recall@3 为 56/56，引用正确为 164/165，no-match 为
8/8。第二套全新独立 holdout 的 Target Recall@3 为 40/40，引用正确为 117/120，引用完整
为 120/120，no-match 为 8/8，21 项门禁全部通过。

第一次 holdout 曾因章节定位、语义引用和域外拒答不足而 HOLD；通用修复后重新冻结，再由未接触
首轮正文的新评委只运行一次第二套 holdout。两轮正式运行累计为 2 次，未根据第二轮结果继续
调参。完整记录见 `docs/competition/g3_2_exit_report.md`。
