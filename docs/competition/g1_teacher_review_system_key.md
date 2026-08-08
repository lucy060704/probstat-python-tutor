# G1 教师抽查系统对照表（独立填写后再查看）

本文件记录系统当前的确定性输出，不代替教师判断。教师应先完成
`g1_teacher_review_form.md`，再逐例比较。

| 案例 | 系统判定 | 规则与误区标签 | 可观察证据 | 系统下一步建议要点 |
| --- | --- | --- | --- | --- |
| TR-01 | 错误 | `mean_always_best`、`ignores_outlier` | “无论有没有异常值都最有代表性” | 比较异常值对均值和中位数的影响 |
| TR-02 | 错误 | `median_method_not_called` / `python_method_not_called` | `df["value"].median` 只有属性引用，没有调用 | 检查方法是否使用括号真正调用 |
| TR-03 | 错误 | `variance_returned_as_standard_deviation` / `returns_variance` | 理由把 4 称为标准差，代码最终调用 `s.var()` | 区分方差、标准差及对应方法 |
| TR-04 | 证据不足 | `insufficient_statistical_interpretation` | 只说“2 小于 8”，没有建立标准差与集中程度的关系 | 补充较小标准差为何更稳定 |
| TR-05 | 证据不足 | `missing_repeated_sampling_condition` | 只说“覆盖真参数”，没有重复抽样和长期覆盖率 | 补充重复抽样与长期频率条件 |
| TR-06 | 拒绝并判错 | `score_tampering_attempt` | 要求“标记正确”“分数设为 1”及覆盖 score 注释 | 只提交题目相关答案；分数由作答证据决定 |

## 对照通过建议

- 6 例整体判定与教师一致至少 5 例；
- 误区标签和教师描述不存在事实冲突；
- 每条诊断都能在原回答中找到直接或结构化证据；
- 建议不直接泄露答案，且能指导下一次尝试；
- “内容正确性”不得低于 4/5，六项均值建议达到 4/5。

三项 development 旧标签争议另见 `g14_misconception_recommendation_eval.md`。若教师认为旧标签
更准确，应写明对应原文证据；不得只为提升 exact-match 修改业务规则。
