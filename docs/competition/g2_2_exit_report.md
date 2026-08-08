# G2.2 描述统计单元阶段门报告

## 结论

截至 2026-08-07，G2.2 已完成并通过评委 Agent 第二轮复审：**PASS，P0=0、P1=0、
P2=0**，工程上允许进入 G2.3。教师尚未填写审核表，因此本单元和所有题目、
知识卡仍保持 `pending_teacher_review`，不能对外宣称“教师审核通过”。

## 可验收产出

| 要求 | 当前证据 | 结果 |
| --- | --- | --- |
| 6 道综合诊断题 | 均值/中位数和方差/标准差各含概念、Python、解释题 | 通过 |
| 知识节点 | 6 题关联 `ds_center`、`ds_spread`、`ds_outlier_robustness`、`ds_context_interpretation` | 通过 |
| 四级提示 | L1/L2 不直接给答案，L3 给中间步骤，L4 为概念/计算/Python/情境四字段 | 通过 |
| 中文作答体验 | 保留内部稳定 slug，用受控 `accepted_answers` 支持“中位数”、“B组”、“A班”等 | 通过 |
| Python 题契约 | 题干要求自行写表达式，`python_code_required=true` | 通过 |
| Python 安全与准确性 | 只做 AST 静态检查；目标变量、列、属性和方法按 Python 大小写精确匹配 | 通过 |
| 统计内容边界 | 同时说明总体/样本方差，Pandas/NumPy `ddof` 默认差异，不做因果外推 | 通过 |
| 2 张原创知识卡 | `mean_median.yaml`、`variance_std.yaml` 经 manifest 和 SHA-256 完整性检查 | 通过 |
| 教师审核包 | `g2_2_teacher_review_form.md` 含数值、API、口径、提示和图谱核对项 | 待人类完成 |

## 关键设计决定

1. **中文别名不改写内部标准答案。** `accepted_answers` 只允许受控、可审计的同义输入，
   避免使用模糊语义判分，也不破坏现有评测 slug。
2. **Python 题从“读现成代码”改为“自行写表达式”。** 学习者须同时提交数值和代码文本；
   代码仅被 `ast.parse` 解析，永不执行。
3. **结构检查与题设数据精确连接。** 中位数只接受 `df["value"].median()` 或
   `df.value.median()`；样本标准差只接受 `s.std()` 或 `np.std(s, ddof=1)`。
   错误列、变量、大小写、参数和自由度都不能误通过。
4. **表面答案正确不覆盖反向理由。** “中位数更受极端值影响”和“小标准差更分散”
   会输出具体 `contradicts` finding，不会被宽泛支持词覆盖。
5. **图谱表达可教关系。** 新增中心→稳健性、离散→情境解释、稳健性→情境解释三条
   `supports` 连边，`ds_center` 明确覆盖 Python 能力。

## 自动化结果

主 Agent 最终验证：

- G2.2/AST/多证据聚焦测试：`48 passed`；
- 全部非 blind pytest：`234 passed`；
- Ruff：`All checks passed`；
- `git diff --check`：通过；
- 加载摘要：15 题、8 单元、33 知识节点、21 条连边、22 章映射、5 张 RAG 知识卡；
- development：判题 100%（32/32）、标签 87.50%（28/32）、建议 100%（32/32）；
- v0.1：判题 94.44%（34/36）、标签 61.11%（22/36）、建议 94.44%（34/36）；
- 一级提示泄露和 API 失败均为 0。

评委 Agent 独立复审：

- 首审发现 Python 大小写绕过和两类反向推理会误通过，判定 FAIL（P1=2）；
- 首审同时发现 README/产品规格的待审状态过期（P2=1）；
- 修复后首轮全部反例正确失败，标准代码与正确理由未被误伤；
- 聚焦复审 `132 passed`，全部非 blind `234 passed`；
- 最终 **PASS，P0=0、P1=0、P2=0**。

正式验收明确排除 `tests/test_eval_datasets.py`，未读取、未运行已失效的
`evals/blind`。

## 内容与版权边界

- 两张知识卡为团队原创 YAML，教材 PDF 未进入 RAG、仓库或赛事平台；
- `mean_median.yaml` SHA-256 为
  `ede5e94d9f5894a10c90c9541275aade720e73f1bf99d7e3eac4da20f2691ca7`；
- `variance_std.yaml` SHA-256 为
  `a448eb3c936d7268acef9201006953e1aaf469d6a0f4928d2a081d911d3891c3`；
- 两张卡都不含题库标准答案、rubric、标签或评测内部字段。

## 剩余风险

1. 教师需填写 `g2_2_teacher_review_form.md`，重点核对两种方差口径、`ddof`、稳健性表述和章节映射；
2. 当前只有数据质量和描述统计两个单元达到教师待审工程状态，其余 6 个未完成；
3. development 标签 87.50% 比 G2.1 的 90.62% 低一例，原因是严格拒绝未使用题设变量 `s` 的
   NumPy 硬编码数组后增加了一个代码冲突标签；不应为提高 exact-match 指标放宽题目契约；
4. G2 终验仍需独立评委建立新的未暴露封存集，旧 blind 永久失去正式资格。

## 下一任务

G2.3 将完成“概率规则与 NumPy 随机模拟”单元：产出原创知识卡、至少 3 道概念/Python/
解释综合题、可观察误区规则、四级提示、NumPy 静态代码约束和教师审核包，继续不配置 ADP、
不执行学习者代码、不读取失效 blind。
