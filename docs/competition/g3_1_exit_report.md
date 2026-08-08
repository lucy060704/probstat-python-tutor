# G3.1 本地 RAG 闭环阶段门报告

## 1. 结论

G3.1 独立评委终审：**PASS**。

- P0：0
- P1：0
- P2：0
- 允许进入下一项 G3.2“冻结检索评测集与指标基线”
- 不代表教师已批准内容，也不代表 Recall@3 或引用正确率已经达标

## 2. 本轮完成范围

- 从正式 manifest 安全加载 15 张团队原创 YAML 知识卡；
- 按结构化字段生成 478 个稳定、带版本和 checksum 的切片；
- 使用标准库实现 NFKC、中文二元组、倒排索引和 BM25 式确定性排序；
- 支持 concept 硬过滤、知识节点加分、top-k、每来源上限和渲染后上下文预算；
- 返回来源、版本、切片 ID、原文摘录、内容 checksum、源文件 checksum 和审核状态；
- 将检索状态与引用锁入 `DiagnosticReport`，可选模型不能修改；
- 保持 TutorAgent 恰好 5 个工具，检索由服务端确定性注入；
- 无命中、建索引失败或检索期故障不阻断判题、候选学习状态和推荐；
- 提供无需 API Key、模型、网络或 ADP 的命令行演示。

最终索引指纹：

```text
sha256:5b301d0c26f277080db8ca92a8036cf0e011eead0020a5860cd42779e3c3343e
```

## 3. 渐进披露与安全边界

最终 478 个切片的最低披露等级分布：

| 等级 | 数量 | 允许内容 |
|---:|---:|---|
| L1 | 121 | 学习目标、前置知识、反思问题 |
| L2 | 204 | 概念解释、一般误区、数据解释原则、无表达式的公式含义 |
| L3 | 102 | 公式表达式、Python API、代码和数学运算结构 |
| L4 | 51 | 完整总结 |

正文级分类会保守识别函数/属性/下标调用、裸 API、LaTeX、等式和数学运算，不能仅靠 YAML
字段名绕过披露等级。33 道题在 L1 均有可用检索结果，但不会返回 L2–L4 切片。

索引只接受：

- `project_authored`；
- `project-owned`；
- `answer_leakage_risk=low`；
- 同时允许 `retrieval` 与 `quotation`；
- manifest 登记且位于受限目录内的 YAML。

教材 PDF、题库答案、rubric、评测 ID/标签、推荐规则 ID、完整四级解释和资料内中英提示注入
均不会进入索引。manifest 标题和公开 `LocalRagIndex` 构造器也执行同样的注入、披露、checksum
与稳定 ID 复核；常见否定性安全说明不会被误杀。

## 4. 独立评委证据

- RAG、资料加载、TutorAgent、Service 专项：158/158 通过；
- 显式非 blind 安全清单：549/549 通过；
- Ruff：通过；
- `git diff --check`：通过；
- 15/15 资料 checksum 独立复算一致；
- 478 个切片覆盖 9 个 `ConceptId`、8 个深度单元；
- 相同输入重复构建的顺序、ID 与索引指纹一致；
- 标题、审核状态、正文或策略字段变化会改变索引指纹；
- 引用摘录均来自实际切片，content/source 双 checksum 一致；
- `no_match` 与 `index_unavailable` 均返回空引用；
- 失败降级、上下文预算、公开构造器和中英注入旁路均通过对抗复核；
- TutorAgent 仍恰好暴露 5 个工具；
- CLI 演示显示 15 个来源、478 个切片，不调用模型、网络或 ADP。

评委没有读取或运行 `evals/blind`、失效 blind 或 `tests/test_eval_datasets.py`。主 Agent 曾误用
全量 `pytest`，间接运行了受限评测数据隔离测试；输出只显示汇总断言，没有查看案例正文，也未据
此调整 RAG、grader、题库或标签。为避免把该流程误写成盲测，本报告不把主 Agent 的 562 项全量
结果作为阶段门证据，只采用评委保持隔离的 549 项显式非 blind 清单与 158 项专项结果。

## 5. 回归指标

公开 v0.1 回归保持：

- 确定性判题：34/36，94.44%；
- 误区标签：22/36，61.11%；
- 下一步建议：34/36，94.44%；
- 一级提示泄露：0/3；
- API 失败：0/36。

这些是历史公开基线，不是 G3.2 的检索质量指标。

## 6. 主要变更文件

- `src/probstat_tutor/rag/retrieval.py`
- `src/probstat_tutor/rag/loader.py`
- `src/probstat_tutor/rag/source_schemas.py`
- `src/probstat_tutor/rag/__init__.py`
- `src/probstat_tutor/schemas.py`
- `src/probstat_tutor/config.py`
- `src/probstat_tutor/tutor_agent.py`
- `src/probstat_tutor/service.py`
- `data/rag/manifest.yaml`
- `tests/test_g3_local_rag.py`
- `tests/test_rag_sources.py`
- `scripts/g3_local_rag_demo.py`
- `README.md`
- `docs/product_spec.md`
- `docs/rag_manifest_spec.md`

全程未新增生产依赖，未执行学习者代码，未配置或发布 ADP。

## 7. 已知风险与下一项

1. 15 张知识卡仍为 `pending_teacher_review`；教师签字前不能声称内容获批。
2. v0.1 历史误区标签准确率仍为 61.11%，不能包装成高准确率。
3. 当前“33 题均可命中”是工程覆盖检查，不等于独立检索质量。
4. G3.2 必须新建并冻结不含正式答案的检索评测集，由独立评委计算 Recall@3、引用正确率、
   无结果准确性和披露安全；在冻结前不得调参或宣称达到 90%/95% 门槛。
