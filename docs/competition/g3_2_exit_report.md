# G3.2 冻结检索评测与质量基线阶段门报告

## 1. 结论

截至 2026-08-08，G3.2 最终阶段门：**PASS**。

- 冻结前代码复审：PASS，P0=0、P1=0、P2=0；
- 第二轮全新独立 holdout：21/21 门禁通过；
- 允许进入 G3.3“本地学生主路径、教师匿名汇总与一键启动”；
- 这证明当前冻结版本通过了本地检索工程门，不等于内容已获教师签字，也不保证真实教学效果。

## 2. 冻结对象

| 对象 | 指纹 |
|---|---|
| development 数据集 | `sha256:b84486116e74d50e4e9152ab65af8893c51f56d3bdb53013ffe2c0285f69490a` |
| 本地索引 | `sha256:5b301d0c26f277080db8ca92a8036cf0e011eead0020a5860cd42779e3c3343e` |
| 检索器 | `sha256:5e0d2f01c151204cdfe7d3e31c3754f201ae6f9fae4d550c8431e8dca9805371` |
| 评测器 | `sha256:9a59205a1a1adf7ae2fbf29e1790ab8acc46ea24d7e0f20113fbea07f0d5cd7a` |
| 最终 holdout | `sha256:a853ebf18ac83bf5b68712c1789bc4dcd340e1c4bc092bcb2793bcbdb10aec09` |

Runner 在计算指标前强制复核上述代码/数据指纹、15 个来源的版本与 checksum，以及
development/holdout 的查询家族、规范化文本、完整查询合同和中文 4-gram 独立性。

## 3. Development 基线

可见 development 共 64 条：56 matched、8 no-match，8 个单元各 7 条 matched；覆盖
9 个 `ConceptId`、15 个来源、三种用途、四个披露等级和 8 个受保护 L1 案例。

| 指标 | 结果 |
|---|---:|
| Target Recall@1 | 46/56 = 82.14% |
| Target Recall@3 | 56/56 = 100% |
| 每单元 Target Recall@3 | 8 个单元均 7/7 |
| Query-only Target Recall@3 | 8/8 = 100% |
| MRR | 0.9107 |
| Citation correctness | 164/165 = 99.39% |
| Citation integrity | 165/165 = 100% |
| No-result accuracy | 8/8 = 100% |
| 披露违规 / L1 泄漏 | 0/64 / 0/24 |
| 三次重放稳定 / 评测失败 | 64/64 / 0/64 |

## 4. 两次独立 holdout 的诚实记录

### 4.1 第一次：HOLD

首个独立评委在旧检索器指纹 `sha256:aa71c4...` 上一次运行 48 条封存集，结果为 HOLD：

- Target Recall@1 27/40 = 67.5%；
- Target Recall@3 35/40 = 87.5%；
- MRR 0.775；
- Citation correctness 88/111 = 79.28%；
- No-result accuracy 5/8 = 62.5%；
- 来源 Recall@3 40/40，但存在“来源正确、章节错位”、语义引用混入和 3 个域外误匹配；
- 共 6 项门禁未通过；只运行一次，未按逐例结果调参，临时材料随后删除。

主 Agent 只收到上述聚合失败类别，没有读取查询、标签、失败 ID 或引用正文。修复严格限制为
节点范围、来源配额、章节意图、课程领域锚点和复合意图覆盖等通用规则，没有修改 BM25 参数、
全局 `top_k`、正式题答案或评测门槛。

### 4.2 第二次：PASS

代码重新冻结后，由未接触首轮正文的全新评委 Agent 独立撰写另一套 48 条 holdout；正式检索
恰好运行一次，结果如下：

| 指标 | 结果 |
|---|---:|
| Target Recall@1 | 32/40 = 80.0% |
| Target Recall@3 | 40/40 = 100% |
| 每单元 Target Recall@3 | 8 个单元均 5/5 |
| Query-only Target Recall@3 | 8/8 = 100% |
| Source Recall@1 / @3 | 39/40 / 40/40 |
| MRR | 0.8875 |
| Relevant-source precision | 117/120 = 97.5% |
| Citation correctness | 117/120 = 97.5% |
| Citation integrity | 120/120 = 100% |
| No-result accuracy | 8/8 = 100% |
| 披露违规 / L1 泄漏 | 0/48 / 0/16 |
| 三次重放稳定 / 评测失败 | 48/48 / 0/48 |
| 平均 / p95 延迟 | 1.196 ms / 8.296 ms |

21 项门禁全部通过。剩余聚合误差是 8 个 Top-1 严格章节错位、1 个 Top-1 来源错位和
3/120 个语义不可接受引用；它们均未突破 Top-3、引用正确率或来源精度门槛。运行后没有调参、
改题、改标签或重跑，临时目录已删除。两轮 holdout 的正式运行累计为 2 次。

## 5. 最终检索策略

- concept 仍是硬过滤；有知识节点且存在合规重叠候选时过滤非重叠候选，无重叠时安全回退；
- 唯一节点来源每来源最多 3 条，多来源或 query-only 每来源最多 2 条；
- 总结、Python、公式采用明确强意图；误区、解释、前置、反思和概念采用较小的通用意图；
- 复合明确意图在 Top-k 内保留章节覆盖，再按原始分数补齐并恢复确定性排序；
- query-only 必须出现 1 个课程特异锚点或至少 2 个一般统计锚点，并继续满足原词法阈值；
- 所有命中仍受披露等级、上下文预算、原创/授权、精确引用和双 checksum 约束。

## 6. 工程验证

- 公开 G3 专项：83/83 通过；
- 扩大后的显式安全回归：215/215 通过；
- Ruff：通过；
- `git diff --check`：通过；
- Development 21 项门禁全部通过；
- 冻结前只读复审：PASS，P0/P1/P2 均为 0；
- 未运行旧 blind 或 `tests/test_eval_datasets.py`；
- 未调用模型、网络或 ADP，未执行学习者代码，未新增生产依赖。

## 7. 主要变更文件

- `src/probstat_tutor/rag/retrieval.py`
- `evals/rag_eval.py`
- `evals/rag/development.jsonl`
- `evals/rag/development_manifest.json`
- `tests/test_g3_rag_eval.py`
- `tests/test_g3_local_rag.py`
- `docs/rag_eval_spec.md`
- `README.md`

## 8. 剩余风险与下一项

1. Agent Team 的共享环境隔离是流程隔离，不是密码学盲测；正式报奖前仍应由教师、队友设备
   或 CI secret 环境复核一次冻结版本。
2. 15 张知识卡继续保持 `pending_teacher_review`，不能宣称内容已获教师批准。
3. Top-1 仍有 8/40 严格章节错位；当前满足既定门槛，但后续展示应以 Top-3 引用为准，不应
   包装成“所有第一条都正确”。
4. G3.3 只做本地产品主路径、教师匿名汇总、一键启动、故障日志和断网降级，不配置或发布 ADP。
