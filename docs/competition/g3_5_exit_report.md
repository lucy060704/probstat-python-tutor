# G3.5 可靠性与故障注入阶段门最终报告

## 1. 主 Agent 结论

截至 2026-08-08，G3.5 最终阶段门：**PASS**。独立评委 Agent 确认 P0=0、P1=0、P2=0，
允许进入 G3.6。

本轮为诊断回执增加请求指纹和兼容迁移；为可选模型增加硬超时、有限重试与进程内熔断；建立
平台无关客户端重试策略；补齐 OpenAPI 请求头、HEAD 和精确状态码；完成无网络故障演示和本机
P50/P95 基线。确定性判题、RAG、掌握度与 SQLite 提交仍是原有单一业务路径，没有复制到 API。

本轮没有配置、调用或发布 ADP，没有访问真实模型或外部网络，没有上传教材 PDF，没有执行
学习者 Python 文本，没有运行旧 blind/holdout，也没有修改冻结 G3.2 检索器或评测器。

## 2. 验收矩阵

| G3.5 要求 | 实现与证据 | 主 Agent 结果 |
| --- | --- | --- |
| 同键同正文重放 | 回执键命中后核对请求指纹并返回原报告 | 200，历史 1 条 |
| 同键改正文冲突 | 首次读取和最终短事务均核对 SHA-256 指纹 | 409，历史仍 1 条 |
| 并发竞争安全 | SQLite `BEGIN IMMEDIATE` 内比较指纹；线程竞争 1 次创建、1 次冲突 | 通过 |
| 旧数据库兼容 | 初始化时只增可空 `request_fingerprint` 列，不删除或猜测旧外部正文 | 通过 |
| 有界客户端重试 | 只重试 retryable 500/503；POST 必须有原幂等键；总尝试最多 3 次 | 通过 |
| 模型硬超时/重试 | 每次 `asyncio.wait_for`；总尝试 1–3；只包围内存解释和 Pydantic 校验 | 通过 |
| 熔断与恢复 | 连续逻辑失败达到阈值后开启；冷却后单个半开探测；成功清零 | 通过 |
| 离线降级 | 超时、耗尽和熔断都返回确定性报告；模型不决定分数或写数据库 | 通过 |
| 存储故障回滚 | 注入 SQLite 回执触发器失败；状态和回执同一事务回滚 | 503，历史 0 条 |
| API 契约精确性 | OpenAPI 覆盖可选 `X-Request-ID`、GET/HEAD/POST 和逐端点状态码 | 通过 |
| 本机性能基线 | 24 个隔离档案、并发上限 6、完整诊断、P95 门槛 3,000 ms | 24/24，通过 |

## 3. 幂等与事务设计

新回执同时持久化：

- `submission_key`：定位匿名档案内的幂等操作；
- `request_fingerprint`：覆盖匿名档案、题目、答案、思考过程、Python 文本和提示层级；
- `report_json`：首次原子提交的诊断报告。

相同正文允许更换请求 ID 和会话 ID 后重放，因为它们不改变逻辑学习提交。不同正文使用同一
外部键时，服务层在调用 TutorAgent 前拒绝已知冲突；若两个请求同时越过首次读取，存储事务会
在任何状态写入前再次判断，返回 `IDEMPOTENCY_CONFLICT`。测试覆盖顺序冲突和两个 SQLite
连接的真实线程竞争。

旧表通过 `ALTER TABLE ... ADD COLUMN` 原位迁移。没有指纹的旧内容哈希回执仍可确认并重放；
无法确认正文的旧外部幂等键保守返回冲突，不猜测、不返回可能属于其他正文的报告。

## 4. 超时、重试与熔断边界

`ModelReliabilityController` 只包围 `Runner.run` 和模型输出 Pydantic 校验。默认每次尝试 8 秒、
一个逻辑解释最多 2 次；代码硬限制最多 3 次。两次尝试之间使用有界线性等待。每个逻辑调用只在
全部尝试失败后增加一次连续失败计数，成功立即清零。

达到阈值后熔断在当前 Python 进程内开启；冷却结束只允许一个半开探测。开启或半开期间健康
检查返回 `optional_model_status=degraded`，不返回模型名、异常、计数或密钥。进程内熔断不是
分布式协调，重启会重置；当前本地单机范围如实保留这一限制。

未来客户端使用 `BoundedApiRetryPolicy`：只对错误体明确标记可重试的 500/503 做有上限指数
等待，总尝试不超过 3，POST 无幂等键不重试。演示注入第一次 503 后复用相同键，第二次 200，
学习历史只有 1 条；409 和 422 不自动重试。

## 5. 隐私与故障日志

- 请求指纹只保存 SHA-256，不新增第二份学习者原文；
- 错误响应不含冲突正文、异常消息、SQLite 触发器文本、模型提供商细节或环境；
- 故障事件仍只有 6 个允许字段；新增稳定码为 `model_timeout`、
  `model_retry_exhausted` 和 `model_circuit_open`；
- 模型重试不包含 SQLite 提交；状态和回执只在最终确定性/增强报告准备完毕后写一次；
- 已识别的不安全提交继续完全隔离于模型，离线模式从不调用模型；
- 学习者 Python 仍只进入 AST 静态检查和安全文本摘录，不编译、不导入、不执行。

## 6. 无网络演示与本机性能结果

执行：

```bash
.venv/bin/python scripts/g3_reliability_demo.py \
  --output docs/competition/g3_5_reliability_result.json
```

环境与结果：

- Python 3.11.15，Darwin arm64，10 个逻辑 CPU；
- HTTPX ASGI 进程内传输，临时 SQLite，无套接字、ADP 或真实模型；
- 同正文重放 200；改正文 409；冲突后历史 1；
- 有界客户端重试 2 次后成功，历史 1，内部故障文本未泄露；
- 回执插入失败 503，历史 0；
- 两个逻辑模型超时共执行 4 次尝试，第三个逻辑请求由熔断直接回退；
- 3/3 模型故障请求均返回并保存确定性本地报告；
- 24/24 隔离诊断成功，每个档案恰有 1 条历史；
- P50 1.573 ms，P95 1.829 ms，最大 1.972 ms，低于本机门槛 3,000 ms；
- `serious_fault_count=0`，`passed=true`。

并发 API 样本由 `asyncio.gather` 调度并限制最多 6 个在途调用；SQLite 的真正并行竞争另由两个
连接的线程测试覆盖。延迟只用于当前机器回归，不代表公网、ADP 或真实模型 SLA。

## 7. 工程验证

- `ruff check .`：通过；
- `git diff --check`：通过；
- `pip check`：`No broken requirements found`；
- `scripts/export_api_contract.py --check`：通过；
- 可靠性/存储/服务/API/产品聚焦：101/101 通过；
- `pytest -q`：664/664 通过，用时 72.20 秒；
- 独立评委聚焦：77/77；最终可靠性子集：20/20；
- 独立评委额外禁止 socket 后重跑演示、用两个 Service/SQLite 连接做线程竞争、注入尝试写文件的
  学习者 Python 文本，并核对超时取消和半开单探测；全部通过；最终 PASS，P0/P1/P2 均为 0；
- G3.2 冻结哈希保持：
  - `retrieval.py`：`5e0d2f01c151204cdfe7d3e31c3754f201ae6f9fae4d550c8431e8dca9805371`；
  - `rag_eval.py`：`9a59205a1a1adf7ae2fbf29e1790ab8acc46ea24d7e0f20113fbea07f0d5cd7a`。

## 8. 主要变更文件

- `.env.example`
- `src/probstat_tutor/reliability.py`
- `src/probstat_tutor/config.py`
- `src/probstat_tutor/observability.py`
- `src/probstat_tutor/tutor_agent.py`
- `src/probstat_tutor/storage.py`
- `src/probstat_tutor/service.py`
- `src/probstat_tutor/api/retry.py`
- `src/probstat_tutor/api/schemas.py`
- `src/probstat_tutor/api/adapter.py`
- `src/probstat_tutor/api/openapi.py`
- `scripts/g3_reliability_demo.py`
- `tests/test_reliability.py`
- `tests/test_api_retry.py`
- `tests/test_storage.py`
- `tests/test_service.py`
- `tests/test_tutor_agent.py`
- `tests/test_api_contract.py`
- `tests/test_g3_reliability_demo.py`
- `docs/api/openapi.json`
- `docs/api_contract.md`
- `docs/reliability.md`
- `docs/competition/g3_5_reliability_result.json`
- `README.md`
- `docs/product_spec.md`

## 9. 剩余风险与下一项

1. 熔断是单进程内存状态，不跨进程共享；当前本地范围不需要伪造分布式协调。
2. SQLite 仍是单机未加密存储；没有生产级备份、密钥托管、数据保留或安全删除流程。
3. 本机 24 请求基线样本较小且不含真实网络/模型；不能宣称线上 SLA 或大规模并发能力。
4. API 仍无公网 TLS、鉴权、授权和限流，绝不能直接公开部署。
5. 15 张原创知识卡仍为 `pending_teacher_review`，工程通过不等于教师内容批准。
6. 下一项为 G3.6：G3 总验收、交付封装和最终要求审计；仍不配置 ADP。
