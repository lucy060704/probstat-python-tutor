# RAG 课程资料 Manifest 规范

## 1. 目的和边界

Manifest 是 RAG 课程资料的登记表。它回答以下问题：

- 资料是谁创建或提供的；
- 资料属于哪个版本；
- 覆盖哪些课程知识点；
- 文件在项目中的什么位置；
- 文件内容是否发生变化；
- 项目被允许如何使用资料；
- 资料是否可能泄露题目答案。

本规范定义资料元数据、课程正文、安全加载与完整性校验，并记录 G3.1 本地确定性检索边界。

当前实现位于 `src/probstat_tutor/rag/schemas.py`，使用 Pydantic 校验。
课程正文合同和加载器分别位于 `src/probstat_tutor/rag/source_schemas.py` 与
`src/probstat_tutor/rag/loader.py`。切片、索引和检索位于
`src/probstat_tutor/rag/retrieval.py`。`data/rag/manifest.example.yaml` 仍是无正文的
格式示例，其中 checksum 是占位值；`data/rag/manifest.yaml` 是正式登记表，其中
checksum 来自 15 份团队原创课程资料的真实文件字节。

## 2. 顶层结构

```yaml
manifest_version: "1.0"
sources:
  - source_id: example_source
    title: 示例资料
    source_type: project_authored
    version: "0.1.0"
    language: zh-CN
    concept_ids: [mean_median]
    file_path: data/rag/sources/example.yaml
    checksum: sha256:0000000000000000000000000000000000000000000000000000000000000001
    license: project-owned
    allowed_usage: [retrieval, quotation]
    answer_leakage_risk: low
    metadata:
      placeholder: true
    updated_at: 2026-07-26T09:00:00+08:00
```

当前只支持 `manifest_version: "1.0"`。未来如果字段含义发生不兼容变化，应增加新的
manifest 版本和迁移说明，不能静默改变 1.0 的含义。

## 3. Source 字段

| 字段 | 必填 | 约束和作用 |
|---|---:|---|
| `source_id` | 是 | 3～64 个字符；小写字母开头，只能包含小写字母、数字和下划线；整个 manifest 中唯一 |
| `title` | 是 | 人类可读标题，1～200 个字符 |
| `source_type` | 是 | 资料来源类型，使用明确枚举 |
| `version` | 是 | 资料版本，格式为 `主版本.次版本.修订版本`，例如 `0.1.0` |
| `language` | 是 | 资料主要语言，使用明确枚举 |
| `concept_ids` | 是 | 至少一个当前课程支持的知识点，不能重复 |
| `file_path` | 是 | 相对于项目根目录的 POSIX 路径，且必须位于 `data/rag/sources/` |
| `checksum` | 是 | `sha256:` 加 64 个小写十六进制字符 |
| `license` | 是 | 项目允许接受的许可证或授权状态 |
| `allowed_usage` | 是 | 至少一种明确允许的用途，不能重复 |
| `answer_leakage_risk` | 是 | 答案泄露风险等级 |
| `metadata` | 是 | 可扩展元数据，默认空对象；键使用小写字母、数字和下划线 |
| `updated_at` | 是 | 最后审核或更新的时间，必须包含时区 |

## 4. 枚举值

### source_type

| 值 | 含义 |
|---|---|
| `project_authored` | 项目成员原创 |
| `open_licensed` | 来自开放许可证资料 |
| `institutional` | 经学校、课程组或机构授权 |
| `public_domain` | 已确认属于公有领域 |

资料来自互联网不代表它是 `public_domain` 或 `open_licensed`。登记前必须核对授权。

### language

- `zh-CN`：简体中文；
- `en`：英语；
- `zh-CN+en`：简体中文与英语混合。

如果以后需要其他语言，应升级 schema，而不是随意写入未定义代码。

### license

- `project-owned`
- `CC-BY-4.0`
- `CC-BY-SA-4.0`
- `CC0-1.0`
- `public-domain`
- `permission-required`

`permission-required` 表示授权尚未完成，不能进入比赛本地检索索引。

### allowed_usage

| 值 | 含义 |
|---|---|
| `retrieval` | 允许进入检索候选 |
| `quotation` | 允许在反馈中短引用 |
| `adaptation` | 允许改写为项目课程内容 |
| `redistribution` | 允许随项目重新分发 |
| `evaluation` | 允许用于离线评测 |
| `audit_only` | 只允许人工安全或授权审查 |

这些值表示项目已经确认的用途，不应根据技术上“能够使用”而自动增加。

### answer_leakage_risk

| 值 | 含义 |
|---|---|
| `low` | 通用概念或背景知识，通常不包含题目答案 |
| `medium` | 含公式、方法或解释，可能间接暴露解题路径 |
| `high` | 含完整例题、详细步骤或接近标准答案的内容 |
| `prohibited` | 含隐藏标准答案、rubric 或评测标签，不能进入普通检索 |

`prohibited` 资料只能设置 `allowed_usage: [audit_only]`。

## 5. 路径安全

`file_path` 的基准是项目根目录，必须满足：

- 只能使用相对路径；
- 必须使用 `/`，不能使用 Windows 反斜杠；
- 不能包含盘符、URL、`.` 或 `..`；
- 必须以 `data/rag/sources/` 开头；
- 必须指向文件，不能只填写目录。

合法示例：

```text
data/rag/sources/mean_median.yaml
```

非法示例：

```text
C:/private/source.yaml
../private/source.yaml
data/rag/../questions.yaml
https://example.com/source.yaml
```

这些规则阻止常见路径穿越。任务 3.2 的加载器还会在读取文件前对候选路径执行
`Path.resolve()`，并检查解析后的路径仍在严格限定的 `data/rag/sources/` 中。通过
符号链接指向该目录之外也会被拒绝；仅仅“仍在项目根目录内”并不足够安全。

## 6. Checksum 和版本管理

真实资料创建后，应对文件原始字节计算 SHA-256：

```text
checksum = "sha256:" + sha256(file_bytes).hexdigest()
```

建议版本规则：

- 修正文案但不改变含义：增加修订版本，例如 `0.1.0` → `0.1.1`；
- 增加或调整课程内容：增加次版本，例如 `0.1.1` → `0.2.0`；
- 字段或课程含义发生不兼容变化：增加主版本。

`updated_at` 必须包含时区。它用于审计，不应替代 checksum；是否变化以文件内容指纹为准。

## 7. 课程正文结构

每份课程资料使用独立 YAML 文件，必须通过 `RagSourceDocument` 校验。统一字段如下：

| 字段 | 作用 |
|---|---|
| `source_schema_version` | 正文数据合同版本，当前只支持 `1.0` |
| `source_id`、`version` | 与 manifest 双向核对资料身份和版本 |
| `title`、`language`、`concept_id` | 标题、语言和唯一课程知识点 |
| `learning_objectives` | 学完后应能完成的初学者目标 |
| `prerequisite_knowledge` | 阅读前需要知道的基础 |
| `concept_explanation` | 概念说明，不针对某道题给答案 |
| `formula_explanation` | 公式、符号、含义、假设和注意事项 |
| `python_connection` | Python 库、API 用途、输入要求和解释提醒 |
| `data_interpretation_guidance` | 从数值走向统计解释时的原则 |
| `common_misconceptions` | 一般性误区、原因和可继续思考的问题 |
| `reflective_questions` | 不带标准答案的反思问题 |
| `summary` | 简短复习要点 |

正文禁止包含题库或评测内部字段，例如 `expected_answer`、`correct_answer`、`ground_truth`、
`case_id`、`eval_case_id`、`numeric_tolerance`、`rubric`、`dimension_weights`、
`grader_findings`、`misconception_tag(s)`、`recommendation_rule_id` 和任何以 `expected_`
开头的字段。试图要求系统忽略规则、泄露答案或修改分数的资料文本也会被拒绝。
加载器还会拒绝与当前题库完整题干完全相同的正文文本。这些检查用于维持课程资料与题库、
评测数据之间的边界，但不能代替人工内容审查。

## 8. 安全加载接口和顺序

使用：

```python
from pathlib import Path

from probstat_tutor.rag import load_rag_manifest, load_rag_source

project_root = Path.cwd()
manifest = load_rag_manifest(project_root / "data/rag/manifest.yaml")
loaded = load_rag_source(manifest.sources[0], project_root)
```

`load_rag_source(manifest_entry, project_root) -> LoadedRagSource` 按以下顺序工作：

1. 验证 manifest entry，并先分类明显的绝对路径和 `..` 路径；
2. 解析真实项目根目录；
3. 解析严格允许的 `data/rag/sources/` 目录；
4. 构造候选路径、检查路径组成部分是否为符号链接；
5. 对候选路径执行 `Path.resolve()`；
6. 验证解析结果仍位于允许目录内；
7. 确认目标存在且为普通文件；
8. 读取真实字节并计算 SHA-256；
9. 对照 manifest 中的 checksum；
10. 按 UTF-8 解码并使用 `yaml.safe_load` 解析；
11. 检查禁用字段、资料内指令注入和完整题干复制；
12. 通过 `RagSourceDocument` schema；
13. 双向核对 `source_id`、`version`、`concept_id` 和 `language`；
14. 根据用途、授权和泄露风险计算切片资格；
15. 返回结构化的 `LoadedRagSource`。

因此，路径安全检查发生在读取文件内容之前；checksum、YAML 和正文身份检查发生在读取之后。

错误通过 `RagSourceLoadError.code` 区分，主要包括：

- 文件不存在或无法读取；
- 绝对路径、`..` 路径穿越、解析后逃逸或符号链接逃逸；
- 目标不是普通文件；
- checksum 不一致或 UTF-8 无效；
- YAML 格式损坏或正文 schema 无效；
- `source_id`、`version`、`concept_id` 或 `language` 不一致；
- 课程正文含题库或评测内部字段。

错误信息使用适合初学者理解的中文，并只显示 manifest 中登记的相对路径，不主动输出
宿主机上的额外敏感路径。

## 9. 切片资格判断

安全加载成功不等于资料一定可以进入后续流程。`eligibility` 包含：

- `eligible_for_chunking: bool`；
- `rejection_reasons: list[str]`，供人阅读；
- `rejection_codes: list[EligibilityRejectionCode]`，供程序稳定判断。

同时满足以下条件时才返回 `true`：

- `allowed_usage` 包含 `retrieval`；
- `license` 不是 `permission-required`；
- `answer_leakage_risk` 为 `low` 或 `medium`。

否则文件仍会完成路径、checksum、YAML、schema 和身份校验，然后返回
`eligible_for_chunking=false` 及明确原因。`high` 或 `prohibited` 风险资料不会静默进入
后续流程，也不会伪装成“文件加载失败”。

这是通用加载资格；比赛正式 `LocalRagIndex` 还会再次收紧为：

- `source_type=project_authored`；
- `license=project-owned`；
- `answer_leakage_risk=low`；
- `allowed_usage` 同时包含 `retrieval` 和 `quotation`；
- 资料必须是 manifest 登记且位于受限目录内的 YAML。

## 10. G3.1 当前能力与边界

当前实现可以证明：

- 资料格式合法；
- manifest 与正文身份一致；
- 文件内容与登记的 checksum 一致；
- 路径被限制在专用资料目录内；
- 系统能判断资料是否有资格进入比赛本地索引；
- 15 张登记知识卡可重建为 478 个稳定切片和一个稳定索引指纹；
- 检索结果包含实际切片、来源版本、精确摘录和双层 checksum；
- 检索遵守知识点过滤、知识节点重排、top-k、每来源上限和上下文预算；
- 提示等级限制公式表达式、Python 连接和总结的披露时机；
- 无匹配或索引不可用时显式降级，不编造引用，也不阻断判题与学习状态更新。

当前仍没有证明 RAG 改善了误区识别或教学效果；Recall@3、引用正确率和真实学习者效果属于
后续冻结评测与试点任务。15 张资料仍为 `pending_teacher_review`，不能写成教师已批准。

## 11. 本地检索接口

```python
from probstat_tutor.config import PROJECT_ROOT
from probstat_tutor.rag import RagQuery, build_local_rag_index
from probstat_tutor.schemas import ConceptId

index = build_local_rag_index(PROJECT_ROOT)
result = index.search(
    RagQuery(
        text="均值 中位数 异常值",
        concept_id=ConceptId.MEAN_MEDIAN,
        disclosure_level=1,
    )
)
```

该接口只使用标准库和已有 Pydantic/YAML 依赖，不使用 Embedding、向量数据库、网络或模型。
`TutorAgent` 在确定性判题后从题目标题、题干、知识点和知识节点构造查询；不把标准答案、rubric、
误区标签、规则 ID 或学习者代码放入查询。检索不是第六个模型工具，诊断中的状态和引用由服务端
锁定。命令行演示见 `scripts/g3_local_rag_demo.py`。
