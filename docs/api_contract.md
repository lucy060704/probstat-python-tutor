# 平台无关学习诊断 API 契约（G3.4–G3.5）

## 1. 范围与传输边界

当前接口是本地验证的 ASGI 适配层，复用 `LearningService`，不复制判题、掌握度、RAG、推荐或
事务逻辑。它的目标是让后期赛事平台只需要适配稳定 JSON/HTTPS 契约，而不会反向修改本地核心。

- 本地启动器固定绑定 `127.0.0.1:8765`，不接受 `--host`，不会暴露到局域网或公网；
- 当前本地验证使用明文 HTTP 回环地址；只用于同一台设备调试；
- 后期若获批准部署，必须由受控网关或反向代理终止 TLS，外部只允许 HTTPS；
- 当前没有公网鉴权、限流、TLS 证书和密钥托管，因此 `x-public-deployment-ready=false`；
- 本轮没有配置或调用 ADP，也没有发出外部网络请求。

直接生产依赖为 Starlette（ASGI 路由）和 Uvicorn（本地服务器）；HTTPX 只属于开发/合同测试。
不引入 FastAPI，也不新增第二套业务逻辑。

## 2. 版本与公共路由

所有 JSON 请求和响应使用 `schema_version="1.0.0"`。机器可读契约位于
`docs/api/openapi.json`，由 Pydantic schema 确定性生成。

| 方法 | 路径 | 请求 | 成功响应 | 状态变化 |
| --- | --- | --- | --- | --- |
| GET | `/health` | 无正文；可带安全 `X-Request-ID` | `HealthResponse` | 无 |
| POST | `/v1/recommend` | `RecommendRequest` | `RecommendResponse` | 无 |
| POST | `/v1/hint` | `HintRequest` | `HintResponse` | 无 |
| POST | `/v1/diagnose` | `DiagnoseRequest` | `DiagnoseResponse` | 原子更新学习状态和回执 |

未知路由、错误方法和业务错误都返回相同的 `ApiErrorResponse`，不会退化成 HTML 或异常堆栈。

## 3. 请求元数据

所有 POST 请求必须包含：

- `schema_version`：当前固定为 `1.0.0`；
- `request_id`：8–128 位 ASCII 字母、数字、下划线或连字符，用于一次调用的追踪；
- `idempotency_key`：8–128 位相同字符集，用于逻辑操作的重试；
- 不允许任何未声明字段。

诊断请求还必须使用形如 `api_anon_<16–64 位十六进制>` 或
`local_anon_<16–64 位十六进制>` 的匿名档案标识。姓名、邮箱、学号和自由文本 ID 都会在 API
入口被拒绝，错误响应只返回字段路径，不回显非法值。

同一匿名档案和同一诊断幂等键会映射到同一提交回执；相同正文重试不会再次更新学习历史。
G3.5 的回执同时保存题目、答案、思考过程、Python 文本和提示层级的 SHA-256 请求指纹，不保存
第二份学习者原文。同一幂等键若改用不同正文，会返回 `409/state_conflict`，不返回第一次报告，
也不改变学习状态。G3.5 以前没有指纹的内容哈希回执可安全兼容；无法确认正文的旧外部键会保守
返回冲突，调用方应生成新键。

可选 `X-Request-ID` 请求头适用于健康检查和正文校验前的错误；有效 POST 请求仍以正文中的
`request_id` 为准。OpenAPI 显式描述该请求头以及 `/health` 的 GET/HEAD 方法。

## 4. 渐进提示与公开题目边界

- `/v1/recommend` 只返回题目 ID、标题、知识点、题型、难度、题干和数据；
- 不返回 `expected_answer`、`accepted_answers`、rubric、grader 规则或四级提示合集；
- `/v1/hint` 每次只返回指定层级；一级不能含完整解释，四级明确标记
  `complete_explanation_revealed=true`；
- `/v1/diagnose` 返回确定性 `DiagnosticReport`；可选模型不能修改分数、证据、引用或推荐；
- 学习者 Python 文本仍只做 AST 静态分析，不编译、不导入、不执行。

## 5. 健康检查

`GET /health` 只返回：

- schema 与服务版本；
- 服务是否可用；
- 确定性离线核心始终可用；
- 可选模型解释是启用还是禁用；
- 本地原创知识索引是否可用。

它不返回 API Key、模型名称、环境变量、数据库路径、账号、机器信息或异常详情。

## 6. 错误合同

| HTTP | 错误码 | 是否建议原请求重试 |
| ---: | --- | --- |
| 400 | `invalid_json` | 否 |
| 404 | `not_found` | 否 |
| 405 | `method_not_allowed` | 否 |
| 409 | `state_conflict` | 否；客户端应刷新状态 |
| 413 | `payload_too_large` | 否 |
| 415 | `unsupported_media_type` | 否 |
| 422 | `invalid_request` | 否；修正字段 |
| 503 | `service_unavailable` | 是；使用相同幂等键 |
| 500 | `internal_error` | 是；不依据本次调用修改分数 |

请求正文上限为 32 KiB。错误响应只含稳定错误码、简体中文安全文案、是否可重试和无值的字段路径；
不包含非法值、学习者原文、异常消息、堆栈、密钥或环境。

## 7. 有界重试、超时与熔断

- 客户端只重试错误体明确标记 `retryable=true` 的 500/503；POST 没有幂等键时禁止重试；
- `BoundedApiRetryPolicy` 最多允许 3 次总尝试，使用有上限的指数等待；409/422 不自动重试；
- 在线模型默认每次尝试硬超时 8 秒、最多 2 次总尝试；等待和次数均可通过受限环境变量调整；
- 失败达到阈值后进程内熔断，冷却期结束只放行一个半开探测；健康检查显示 `degraded`；
- 模型重试只重复内存中的解释生成，不包含 SQLite 提交，不会重复计分；
- 模型超时、重试耗尽或熔断时返回确定性本地报告，并仅记录不含异常消息的允许字段事件。

完整默认值、故障矩阵和限制见 `docs/reliability.md`。

## 8. 本地命令

只验证依赖和路由，不打开端口：

```bash
.venv/bin/python scripts/run_local_api.py --check
```

启动回环 API：

```bash
.venv/bin/python scripts/run_local_api.py --port 8765
```

无外部网络合同演示：

```bash
.venv/bin/python scripts/g3_api_contract_demo.py \
  --output docs/competition/g3_4_api_contract_result.json
.venv/bin/python scripts/g3_reliability_demo.py \
  --output docs/competition/g3_5_reliability_result.json
```

重新生成并检查 OpenAPI：

```bash
.venv/bin/python scripts/export_api_contract.py
.venv/bin/python scripts/export_api_contract.py --check
```

合同测试使用 HTTPX `ASGITransport` 直接运行完整 ASGI 栈，不打开套接字、不访问 ADP，也不依赖
互联网。G3.5 可靠性演示的并发与 P95 仅是当前机器的离线基线；真实外部 HTTPS、鉴权、限流、
分布式熔断和赛事平台负载测试仍属于后续适配阶段。
